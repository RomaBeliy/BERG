import os
import re

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from berg.core.exceptions import InvalidParameterError
from berg.core.model_registry import register_model
from berg.core.parameter_validator import validate_selection_keys, validate_subjects
from berg.interfaces.base_model import BaseModelInterface
from berg.models.fmri.universal_brain_encoder.encoder_models import Encoder, encoder_param


YAML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model_cards", "fmri-nsd_laion_fmri_fsaverage-universal_brain_encoder.yaml"
)
with open(YAML_PATH, "r") as f:
    model_info = yaml.safe_load(f)

register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.fmri.nsd_laion_fmri_fsaverage_universal_brain_encoder",
    class_name="UniversalBrainEncoder",
    modality="fmri",
    training_dataset="nsd_laion_fmri_fsaverage",
    yaml_path=YAML_PATH,
)

# model folder inside the BERG directory
MODEL_DIR = "encoding_models/modality-fmri/train_dataset-nsd_laion_fmri_fsaverage/model-universal_brain_encoder"
# the checkpoint (trained on NSD + LAION-fMRI), in MODEL_DIR/encoding_models_weights/
CHECKPOINT = "universal_brain_encoder_nsd_laion.pth"
# one metadata file per dataset, in MODEL_DIR/metadata/
METADATA = {
    "NSD": "metadata_nsd.npy",
    "LAION": "metadata_laion.npy",
}
DINOV2_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# other names a group can be selected by
GROUP_ALIASES = {
    "retinotopic": ["visual", "prf-visualrois"],
    "faces": ["floc-faces"],
    "bodies": ["floc-bodies"],
    "places": ["fplaces", "floc-places"],
    "words": ["floc-words"],
}
# group prefixes that may come before an ROI name, e.g. "bodies_EBA"
GROUP_PREFIXES = {"bodies", "faces", "fplaces", "places", "words", "visual", "retinotopic", "streams"}


def roi_names(label):
    # all names that select an ROI, e.g. "FFA-1" -> ffa-1, ffa1, ffa;  "V1d" -> v1d, v1
    names = {label, label.replace("-", ""), re.sub(r"-\d+$", "", label)}
    visual = re.fullmatch(r"h?(V\d)[dv]?", label)  # V1d -> V1, hV4 -> V4
    if visual:
        names.add(visual.group(1))
    return {n.lower() for n in names}


def group_names(label):
    # all names that select a group, e.g. "retinotopic" -> retinotopic, visual, prf-visualrois
    return {n.lower() for n in [label, *GROUP_ALIASES.get(label, [])]}


def load_checkpoint(berg_dir):
    path = os.path.join(berg_dir, MODEL_DIR, "encoding_models_weights", CHECKPOINT)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_metadata(berg_dir, subjects):
    # {subject: its entry in its dataset's metadata file}
    files, metadata = {}, {}
    for s in subjects:
        dataset = s.rsplit("-", 1)[0]
        if dataset not in files:
            path = os.path.join(berg_dir, MODEL_DIR, "metadata", METADATA[dataset])
            files[dataset] = np.load(path, allow_pickle=True).item()
        if s not in files[dataset]:
            raise InvalidParameterError(f"Subject {s} is not in {METADATA[dataset]}")
        metadata[s] = files[dataset][s]
    return metadata


def build_encoder(config):
    param = encoder_param(config["param"]["num_voxels"])
    param.__dict__.update(config["param"])
    backbone = torch.hub.load(
        DINOV2_REPO, config["backbone"], pretrained=False, trust_repo=True, skip_validation=True
    )
    return Encoder(
        param,
        backbone,
        num_embeds=config["num_embeds"],
        select_layers=config["select_layers"],
        r=config["r"],
        alpha=config["alpha"],
        include_reg_tokens=config["include_reg_tokens"],
        num_reg_tokens=config["num_reg_tokens"],
    )


def preprocess(images):
    # images: [B, 3, H, W] (channels first) or [B, H, W, 3] (channels last), values 0-255
    if images.shape[1] != 3 and images.shape[-1] == 3:
        images = np.transpose(images, (0, 3, 1, 2))  # channels last -> channels first
    x = torch.from_numpy(images).float() / 255.0
    # center crop to a square, then resize to 224 x 224
    h, w = x.shape[-2:]
    size = min(h, w)
    top, left = (h - size) // 2, (w - size) // 2
    x = x[..., top:top + size, left:left + size]
    x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class UniversalBrainEncoder(BaseModelInterface):
    MODEL_ID = model_info["model_id"]
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]

    def __init__(self, subject, selection=None, device="auto", berg_dir=None, **kwargs):
        self.subjects = validate_subjects(subject, self.VALID_SUBJECTS)

        selection = selection or {}
        validate_selection_keys(selection, ["roi", "hemisphere"])
        self.rois = selection.get("roi")
        if isinstance(self.rois, str):
            self.rois = [self.rois]
        if self.rois is not None:
            if not all(isinstance(r, str) and r for r in self.rois):
                raise InvalidParameterError(f"roi must be a name or a list of names, got {self.rois}")
            # not case sensitive, and "bodies_EBA" is the same as "EBA"
            self.rois = [r.lower() for r in self.rois]
            self.rois = [r.split("_", 1)[1] if "_" in r and r.split("_", 1)[0] in GROUP_PREFIXES else r
                         for r in self.rois]
        self.hemisphere = selection.get("hemisphere", "both")
        if self.hemisphere not in ["both", "lh", "rh"]:
            raise InvalidParameterError(f"hemisphere must be 'both', 'lh' or 'rh', got {self.hemisphere}")

        self.berg_dir = berg_dir
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = None

    def load_model(self):
        ckpt = load_checkpoint(self.berg_dir)
        self.config = ckpt["config"]

        missing = [s for s in self.subjects if s not in self.config["subjects"]]
        if missing:
            raise InvalidParameterError(f"Subjects {missing} are not in the checkpoint")

        # each subject's voxels are a slice of the shared voxel-embedding table
        num_voxels = np.array(self.config["num_voxels_subjects"])
        start = np.cumsum(num_voxels) - num_voxels

        # select the voxels first (from the metadata files), so selection errors come before building the model
        self.metadata = load_metadata(self.berg_dir, self.subjects)
        self.selected_voxels = {}  # subject voxel indices
        self.voxel_ind = {}  # indices into the voxel-embedding table
        for s in self.subjects:
            i = self.config["subjects"].index(s)
            assert self.metadata[s]["num_voxels"] == num_voxels[i], \
                f"{s}: {self.metadata[s]['num_voxels']} voxels in the metadata vs {num_voxels[i]} in the checkpoint"
            self.selected_voxels[s] = self.select_voxels(s)
            self.voxel_ind[s] = torch.from_numpy(start[i] + self.selected_voxels[s]).to(self.device)

        self.model = build_encoder(self.config)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    def select_voxels(self, subject):
        # all voxels, or only the voxels of the selected ROIs/groups and/or hemisphere
        meta = self.metadata[subject]
        voxels = np.arange(meta["num_voxels"])

        if self.rois is not None:
            rois, groups = meta["rois"], meta["roi_groups"]
            roi_alias = {label: roi_names(label) for label in set(rois["labels"])}
            group_alias = {label: group_names(label) for label in set(groups["labels"])}
            for r in self.rois:
                if not any(r in names for names in [*roi_alias.values(), *group_alias.values()]):
                    raise InvalidParameterError(f"ROI {r} is not available for {subject}")

            # a voxel can be in several ROIs/groups, so the same voxel can match several names
            in_roi = np.array([any(r in roi_alias[l] for r in self.rois) for l in rois["labels"]], dtype=bool)
            in_group = np.array([any(r in group_alias[l] for r in self.rois) for l in groups["labels"]], dtype=bool)
            voxels = np.unique(np.concatenate([
                np.asarray(rois["inds"], dtype=np.int64)[in_roi],
                np.asarray(groups["inds"], dtype=np.int64)[in_group],
            ]))  # sorted, no duplicates

        if self.hemisphere != "both":
            is_lh = np.asarray(meta["hemis"])[voxels]
            voxels = voxels[is_lh] if self.hemisphere == "lh" else voxels[~is_lh]

        return voxels

    def generate_response(self, stimulus, batch_size=8, **kwargs):
        # returns {dataset: {"sub-XX": [num_images, num_voxels]}}
        responses = {}
        for s, vox_ind in self.voxel_ind.items():
            preds = []
            for i in range(0, len(stimulus), batch_size):
                x = preprocess(stimulus[i:i + batch_size]).to(self.device)
                with torch.no_grad():
                    preds.append(self.model(x, vox_ind.expand(len(x), -1)).cpu().numpy())

            dataset, sub = s.rsplit("-", 1)
            responses.setdefault(dataset, {})[f"sub-{sub}"] = np.concatenate(preds)
        return responses

    @classmethod
    def get_metadata(cls, berg_dir=None, subject=None, model_instance=None, **kwargs):
        if model_instance is not None:
            subjects, metadata = model_instance.subjects, model_instance.metadata
        else:
            subjects = validate_subjects(subject, cls.VALID_SUBJECTS)
            metadata = load_metadata(berg_dir, subjects)

        out = {}
        for s in subjects:
            meta = metadata[s]
            # voxels of the output columns (all voxels when there is no loaded model)
            if model_instance is not None:
                voxels = model_instance.selected_voxels[s]
            else:
                voxels = np.arange(meta["num_voxels"])

            dataset, sub = s.rsplit("-", 1)
            out.setdefault(dataset, {})[f"sub-{sub}"] = {
                "num_voxels": int(meta["num_voxels"]),
                "voxel_indices": voxels,
                "coords": np.asarray(meta["coords"])[voxels],
                "hemis": np.asarray(meta["hemis"])[voxels],
                "vertex_inds": np.asarray(meta["vertex_inds"])[voxels],
            }
        return out

    @classmethod
    def get_model_id(cls):
        return cls.MODEL_ID

    def cleanup(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
