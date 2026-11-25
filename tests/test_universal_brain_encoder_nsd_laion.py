"""Test the BERG setup of the Universal Brain Encoder (NSD + LAION-fMRI) on the real test data.

1. Setup:                files in the BERG layout; checkpoint, model card and metadata agree
2. Each dataset alone:   all voxels, median correlation per subject on its test images
3. Subject combinations: a subject gets the same predictions whatever subjects are loaded with it
4. Voxel combinations:   ROIs, groups, name forms and hemispheres select the voxels listed in the
                         metadata files, with the same predictions as the all-voxel run
5. Metadata without loading the model
6. Errors for bad parameters
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import numpy as np
import torch
from scipy.stats import pearsonr

from berg import BERG
from berg.core.exceptions import InvalidParameterError
from berg.models.fmri.nsd_laion_fmri_fsaverage_universal_brain_encoder import (
    CHECKPOINT, METADATA, MODEL_DIR, UniversalBrainEncoder)

MODEL_ID = "fmri-nsd_laion_fmri_fsaverage-universal_brain_encoder"
BERG_DIR = "/home/romanb/data/BERG/berg_dir"
CHECKPOINT_SRC = "/home/romanb/data/universal-brain-encoder_debug/results/saved_models/universal_brain_encoder_nsd_laion.pth"
NSD_DIR = "/home/romanb/data/datasets/NVD/tutorial_data/subjects/subsuper/"
LAION_DIR = "/home/romanb/data/LAION-fMRI/data/"

NSD_TEST_SUBJECTS = ["NSD-01", "NSD-02", "NSD-05", "NSD-07"]  # subjects with test responses in multi_sub_fmri
ROI_TEST_SUBJECTS = ["NSD-01", "LAION-01"]  # one subject per dataset for the voxel combinations
NUM_VERTICES = 163842  # fsaverage7 vertices per hemisphere
TOL = 1e-3  # max |difference| between two runs that must give the same predictions

results = []


def check(ok, text):
    results.append((bool(ok), text))
    print(f"  {'PASS' if ok else 'FAIL'}  {text}")


def section(title):
    print(f"\n{'=' * 120}\n{title}\n{'=' * 120}")


# ─────────────────────────────────────────────────────────────────────────────
section("1. SETUP")
model_dir = os.path.join(BERG_DIR, MODEL_DIR)
ckpt_path = os.path.join(model_dir, "encoding_models_weights", CHECKPOINT)
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
if not os.path.lexists(ckpt_path):
    os.symlink(CHECKPOINT_SRC, ckpt_path)
paths = [ckpt_path] + [os.path.join(model_dir, "metadata", f) for f in METADATA.values()]
for p in paths:
    check(os.path.isfile(p), f"file in place: {p}")
if not all(os.path.isfile(p) for p in paths):
    raise SystemExit("Missing files, see above")

# checkpoint vs model card
config = torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"]
SUBJECTS = list(config["subjects"])
NUM_VOXELS = dict(zip(SUBJECTS, config["num_voxels_subjects"]))
LAION_SUBJECTS = [s for s in SUBJECTS if s.startswith("LAION-")]
check(sorted(SUBJECTS) == sorted(UniversalBrainEncoder.VALID_SUBJECTS), f"checkpoint subjects = model card subjects: {SUBJECTS}")

# metadata files vs checkpoint
RAW = {}  # the metadata files as stored, {subject: entry}
for fname in METADATA.values():
    RAW.update(np.load(os.path.join(model_dir, "metadata", fname), allow_pickle=True).item())
for s in SUBJECTS:
    m = RAW.get(s)
    if m is None:
        check(False, f"{s}: not in the metadata files")
        continue
    hemis = np.asarray(m["hemis"], dtype=bool)
    ok = m["num_voxels"] == NUM_VOXELS[s] == len(hemis) == len(m["coords"]) == len(m["vertex_inds"])
    check(ok, f"{s:<9} voxels: checkpoint {NUM_VOXELS[s]:>6}, metadata {m['num_voxels']:>6} ({hemis.sum()} lh, {(~hemis).sum()} rh)  "
              f"{len(set(m['rois']['labels']))} ROIs, {len(set(m['roi_groups']['labels']))} groups")

# test data: only the test images and the measured test responses; voxel counts come from the checkpoint
nsd_fmri = np.load(os.path.join(NSD_DIR, "fmri_v2.npz"))
nsd_images = np.load(os.path.join(NSD_DIR, "all_images_v2.npy"), mmap_mode="r")
NSD_IMAGES = np.transpose(nsd_images[nsd_fmri["type_sample"] == 2], (0, 3, 1, 2))
NSD_FMRI = nsd_fmri["multi_sub_fmri"]  # NSD subjects one after the other, in the checkpoint's order
nsd_counts = [NUM_VOXELS[s] for s in SUBJECTS if s.startswith("NSD-")]
NSD_START = dict(zip([s for s in SUBJECTS if s.startswith("NSD-")], np.cumsum(nsd_counts) - nsd_counts))

laion_fmri = np.load(os.path.join(LAION_DIR, "laion_fmri_v2.npz"), allow_pickle=True)
laion_images = np.load(os.path.join(LAION_DIR, "laion_images_224.npy"), mmap_mode="r")
LAION_IMAGES = np.transpose(laion_images[laion_fmri["type_sample"] == 2], (0, 3, 1, 2))
LAION_FMRI = laion_fmri["test_fmri"]
LAION_SUB = laion_fmri["test_sub"]
LAION_INDEX = {"LAION-" + k.split("-")[-1]: v for k, v in laion_fmri["index_mapping"].item()["subject_to_index"].items()}
print(f"\n  test images: NSD {len(NSD_IMAGES)}, LAION { {s: int((LAION_SUB == LAION_INDEX[s]).sum()) for s in LAION_SUBJECTS} }")


def test_rows(s, num_images=None):
    # rows of the subject's test images in NSD_IMAGES / LAION_IMAGES (same rows as its test fMRI)
    rows = np.arange(len(NSD_IMAGES)) if s.startswith("NSD-") else np.flatnonzero(LAION_SUB == LAION_INDEX[s])
    return rows[:num_images]


def measured(s, rows):
    # measured test responses, all the subject's voxels
    if s.startswith("NSD-"):
        return NSD_FMRI[rows, NSD_START[s]:NSD_START[s] + NUM_VOXELS[s]].astype(np.float32)
    return LAION_FMRI[rows, :NUM_VOXELS[s]].astype(np.float32)


def median_r(pred, gt):
    return float(np.median(pearsonr(pred, gt, axis=0)[0]))


berg = BERG(BERG_DIR)


def run(subjects, selection=None, num_images=None):
    # load the model with BERG, predict each subject's test images
    # returns {subject: (prediction, metadata, rows)}
    model = berg.get_encoding_model(MODEL_ID, subject=subjects, selection=selection)
    out = {}
    for dataset, images in [("NSD", NSD_IMAGES), ("LAION", LAION_IMAGES)]:
        subs = [s for s in subjects if s.startswith(dataset + "-")]
        if not subs:
            continue
        rows = {s: test_rows(s, num_images) for s in subs}
        all_rows = np.unique(np.concatenate(list(rows.values())))
        result = berg.encode(model, images[all_rows], return_metadata=True)
        if not isinstance(result, tuple):
            raise RuntimeError("berg.encode did not return the metadata")
        preds, meta = result
        expected = {}
        for s in subjects:
            d, n = s.split("-")
            expected.setdefault(d, set()).add(f"sub-{n}")
        if {d: set(v) for d, v in preds.items()} != expected:
            check(False, f"output subjects { {d: sorted(v) for d, v in preds.items()} }, expected {expected}")
        for s in subs:
            sub = "sub-" + s.split("-")[1]
            out[s] = (preds[dataset][sub][np.searchsorted(all_rows, rows[s])], meta[dataset][sub], rows[s])
    model.cleanup()
    return out


def diff_to_ref(s, pred, meta):
    # max |difference| to the section 2 predictions of the same voxels and images
    ref = REF[s][0][:len(pred)]
    return float(np.abs(pred - ref[:, meta["voxel_indices"]]).max())


# ─────────────────────────────────────────────────────────────────────────────
section("2. EACH DATASET SEPARATELY (all voxels)")
REF = {}  # {subject: (prediction, metadata)}, all voxels, all test images
for name, subjects in [("NSD only", NSD_TEST_SUBJECTS), ("LAION only", LAION_SUBJECTS)]:
    print(f"\n  {name}: {subjects}")
    for s, (pred, meta, rows) in run(subjects).items():
        REF[s] = (pred, meta)
        n = NUM_VOXELS[s]
        hemis = np.asarray(meta["hemis"], dtype=bool)
        vertex_inds, coords = np.asarray(meta["vertex_inds"]), np.asarray(meta["coords"])
        ok = (pred.shape == (len(rows), n)
              and np.array_equal(meta["voxel_indices"], np.arange(n))
              and coords.shape == (n, 3) and len(hemis) == len(vertex_inds) == n
              and vertex_inds.min() >= 0 and vertex_inds.max() < NUM_VERTICES
              and coords[hemis, 0].mean() < 0 < coords[~hemis, 0].mean())  # lh on the left (x < 0)
        check(ok, f"{s:<9} test images {len(rows):>5}  voxels {pred.shape[1]:>6}  median r {median_r(pred, measured(s, rows)):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
section(f"3. SUBJECT COMBINATIONS (max |diff| to section 2 must be < {TOL})")
for subjects in [["NSD-01"], ["LAION-03"], ["NSD-02", "LAION-05"], ["LAION-07", "NSD-05", "LAION-01"]]:
    print(f"\n  subjects: {subjects}")
    for s, (pred, meta, rows) in run(subjects).items():
        d = diff_to_ref(s, pred, meta)
        check(pred.shape == (len(rows), NUM_VOXELS[s]) and d < TOL,
              f"{s:<9} test images {len(rows):>5}  voxels {pred.shape[1]:>6}  "
              f"median r {median_r(pred, measured(s, rows)):.4f}  max |diff| {d:.1e}")

num_images = 16
print(f"\n  all {len(SUBJECTS)} subjects, first {num_images} test images")
for s, (pred, meta, rows) in run(SUBJECTS, num_images=num_images).items():
    d = diff_to_ref(s, pred, meta) if s in REF else None
    check(pred.shape == (len(rows), NUM_VOXELS[s]) and (d is None or d < TOL),
          f"{s:<9} shape {str(pred.shape):<14}" + (f"  max |diff| {d:.1e}" if d is not None else "  (not in section 2)"))

# ─────────────────────────────────────────────────────────────────────────────
section(f"4. VOXEL COMBINATIONS (voxels must match the metadata files, max |diff| to section 2 must be < {TOL})")
FFA = ("rois", ["FFA", "FFA-1", "FFA-2"])
SELECTIONS = [
    # (selection, [(where in the metadata file, labels) for each requested name])
    ({"roi": "FFA-1"}, [("rois", ["FFA-1"])]),
    ({"roi": "ffa1"}, [("rois", ["FFA-1"])]),
    ({"roi": "faces_FFA-1"}, [("rois", ["FFA-1"])]),
    ({"roi": "FFA"}, [FFA]),
    ({"roi": "V1"}, [("rois", ["V1", "V1d", "V1v"])]),
    ({"roi": "V4"}, [("rois", ["V4", "hV4"])]),
    ({"roi": "faces"}, [("roi_groups", ["faces"])]),
    ({"roi": "visual"}, [("roi_groups", ["retinotopic"])]),
    ({"roi": "floc-places"}, [("roi_groups", ["places"])]),
    ({"roi": "MT"}, [("rois", ["MT"])]),
    ({"roi": "motion"}, [("roi_groups", ["motion"])]),
    ({"roi": ["FFA", "PPA", "EBA"]}, [FFA, ("rois", ["PPA"]), ("rois", ["EBA"])]),
    ({"hemisphere": "lh"}, []),
    ({"hemisphere": "rh"}, []),
    ({"roi": "FFA", "hemisphere": "rh"}, [FFA]),
]


def expected_voxels(m, selection, parts):
    # voxels the selection must give, read directly from the metadata file; None if a requested name has no voxels
    voxels = np.arange(m["num_voxels"])
    if parts:
        found = []
        for kind, labels in parts:
            inds, labs = np.asarray(m[kind]["inds"], dtype=np.int64), np.asarray(m[kind]["labels"])
            found.append(inds[np.isin(labs, labels)])
        if any(len(f) == 0 for f in found):
            return None
        voxels = np.unique(np.concatenate(found))
    hemis = np.asarray(m["hemis"], dtype=bool)
    if selection.get("hemisphere") == "lh":
        voxels = voxels[hemis[voxels]]
    if selection.get("hemisphere") == "rh":
        voxels = voxels[~hemis[voxels]]
    return voxels


for s in ROI_TEST_SUBJECTS:
    print(f"\n  {s}")
    for selection, parts in SELECTIONS:
        expected = expected_voxels(RAW[s], selection, parts)
        text = f"{s:<9} {str(selection):<40}"
        try:
            pred, meta, rows = run([s], selection)[s]
        except InvalidParameterError as e:
            check(expected is None, f"{text} not available: {e}")
            continue
        if expected is None:
            check(False, f"{text} expected an error, got {pred.shape[1]} voxels")
            continue
        vox = np.asarray(meta["voxel_indices"])
        same_meta = all(np.array_equal(meta[k], np.asarray(RAW[s][k])[vox]) for k in ["coords", "hemis", "vertex_inds"])
        d = diff_to_ref(s, pred, meta)
        check(np.array_equal(vox, expected) and pred.shape == (len(rows), len(expected)) and same_meta and d < TOL,
              f"{text} voxels {len(vox):>6} (metadata file {len(expected):>6})  "
              f"median r {median_r(pred, measured(s, rows)[:, vox]):.4f}  max |diff| {d:.1e}")

# ─────────────────────────────────────────────────────────────────────────────
section("5. METADATA WITHOUT LOADING THE MODEL (must equal the metadata of the loaded all-voxel model)")
meta_all = berg.get_model_metadata(MODEL_ID, subject=ROI_TEST_SUBJECTS)
for s in ROI_TEST_SUBJECTS:
    d, n = s.split("-")
    m, ref = meta_all[d][f"sub-{n}"], REF[s][1]
    ok = m["num_voxels"] == NUM_VOXELS[s] and all(
        np.array_equal(m[k], ref[k]) for k in ["voxel_indices", "coords", "hemis", "vertex_inds"])
    check(ok, f"{s:<9} voxels {m['num_voxels']:>6}  keys {sorted(m)}")

# ─────────────────────────────────────────────────────────────────────────────
section("6. ERRORS FOR BAD PARAMETERS (each must raise InvalidParameterError)")
for kwargs in [dict(subject="NSD-09"), dict(subject="LAION-02"), dict(subject="sub-01"),
               dict(subject="NSD-01", selection={"roi": "not_an_roi"}),
               dict(subject="NSD-01", selection={"hemisphere": "left"}),
               dict(subject="NSD-01", selection={"bad_key": 1})]:
    try:
        berg.get_encoding_model(MODEL_ID, **kwargs).cleanup()
        check(False, f"{str(kwargs):<60} no error")
    except InvalidParameterError as e:
        check(True, f"{str(kwargs):<60} {str(e)[:90]}")

# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY")
failed = [text for ok, text in results if not ok]
print(f"  {len(results) - len(failed)} / {len(results)} checks passed")
for text in failed:
    print(f"  FAIL  {text}")
