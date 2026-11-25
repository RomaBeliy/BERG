"""Encoding accuracy of the Universal Brain Encoder (NSD + LAION-fMRI), for BERG's encoding_models_accuracy/ folder.

Per subject, on its test images (NSD: the shared test images; LAION-fMRI: the test images of the official tau split):
  - correlation of each voxel: Pearson r between predicted and measured responses
  - median correlation over voxels
  - correlation histogram

Output, in BERG_DIR/<model folder>/encoding_models_accuracy/:
  median_correlation_per_subject.csv   median correlation of each subject
  median_correlation_per_subject.png   the same, all subjects of both datasets in one plot
  correlation_histogram_<subject>.png

Also writes correlations.npz in the folder the script is run from: per-voxel correlations, one array per subject
("NSD-01", ...), in the model's voxel order; used by brain_maps_universal_brain_encoder_nsd_laion.py.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from scipy.stats import pearsonr

from berg import BERG
from berg.models.fmri.nsd_laion_fmri_fsaverage_universal_brain_encoder import CHECKPOINT, MODEL_DIR

MODEL_ID = "fmri-nsd_laion_fmri_fsaverage-universal_brain_encoder"
BERG_DIR = "/home/romanb/data/BERG/berg_dir"
NSD_DIR = "/home/romanb/data/datasets/NVD/tutorial_data/subjects/subsuper/"
LAION_DIR = "/home/romanb/data/LAION-fMRI/data/"
OUT_DIR = os.path.join(BERG_DIR, MODEL_DIR, "encoding_models_accuracy")
CORRELATIONS_PATH = os.path.join(os.getcwd(), "correlations.npz")  # not in OUT_DIR

NSD_SUBJECTS = ["NSD-01", "NSD-02", "NSD-05", "NSD-07"]  # subjects with test responses in multi_sub_fmri
DATASET_COLORS = {"NSD": "#2a78d6", "LAION": "#eb6834"}

# ── Subjects and voxel counts, from the checkpoint ────────────────────────────
ckpt_path = os.path.join(BERG_DIR, MODEL_DIR, "encoding_models_weights", CHECKPOINT)
config = torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"]
NUM_VOXELS = dict(zip(config["subjects"], config["num_voxels_subjects"]))
LAION_SUBJECTS = [s for s in config["subjects"] if s.startswith("LAION-")]
nsd_all = [s for s in config["subjects"] if s.startswith("NSD-")]
nsd_counts = [NUM_VOXELS[s] for s in nsd_all]
NSD_START = dict(zip(nsd_all, np.cumsum(nsd_counts) - nsd_counts))

# ── Test data: only the test images and the measured test responses ───────────
nsd_fmri = np.load(os.path.join(NSD_DIR, "fmri_v2.npz"))
nsd_images = np.load(os.path.join(NSD_DIR, "all_images_v2.npy"), mmap_mode="r")
NSD_IMAGES = np.transpose(nsd_images[nsd_fmri["type_sample"] == 2], (0, 3, 1, 2))
NSD_FMRI = nsd_fmri["multi_sub_fmri"]  # NSD subjects one after the other, in the checkpoint's order

laion_fmri = np.load(os.path.join(LAION_DIR, "laion_fmri_v2.npz"), allow_pickle=True)
laion_images = np.load(os.path.join(LAION_DIR, "laion_images_224.npy"), mmap_mode="r")
LAION_IMAGES = np.transpose(laion_images[laion_fmri["type_sample"] == 2], (0, 3, 1, 2))
LAION_FMRI = laion_fmri["test_fmri"]
LAION_SUB = laion_fmri["test_sub"]
subject_to_index = laion_fmri["index_mapping"].item()["subject_to_index"]  # {"sub-01": 0, ...}
LAION_INDEX = {"LAION-" + k.split("-")[-1]: v for k, v in subject_to_index.items()}


def test_data(subject):
    # the subject's test images [N, 3, H, W] and measured responses [N, voxels]
    if subject.startswith("NSD-"):
        return NSD_IMAGES, NSD_FMRI[:, NSD_START[subject]:NSD_START[subject] + NUM_VOXELS[subject]].astype(np.float32)
    rows = LAION_SUB == LAION_INDEX[subject]
    return LAION_IMAGES[rows], LAION_FMRI[rows, :NUM_VOXELS[subject]].astype(np.float32)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_histogram(corr, subject, median, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(corr, bins=100, color="tab:blue")
    ax.axvline(median, color="black", linestyle="--", label=f"median r = {median:.3f}")
    ax.set_xlabel("Pearson r (predicted vs measured)")
    ax.set_ylabel("Number of voxels")
    ax.set_title(f"{subject} | {len(corr)} voxels")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_medians(medians, path):
    # one bar per subject, colored by dataset
    subjects = list(medians)
    fig, ax = plt.subplots(figsize=(1.5 + 0.7 * len(subjects), 4))
    bars = ax.bar(subjects, [medians[s] for s in subjects], width=0.7,
                  color=[DATASET_COLORS[s.split("-")[0]] for s in subjects])
    ax.bar_label(bars, labels=[f"{medians[s]:.3f}" for s in subjects], padding=2, fontsize=8)
    ax.set_ylabel("Median Pearson r (predicted vs measured)")
    ax.set_title("Median correlation per subject, test images")
    ax.legend(handles=[Patch(color=c, label=d) for d, c in DATASET_COLORS.items()], frameon=False,
              loc="upper left", bbox_to_anchor=(1, 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Accuracy per subject ──────────────────────────────────────────────────────
berg = BERG(BERG_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

correlations, medians, table = {}, {}, []
for subject in NSD_SUBJECTS + LAION_SUBJECTS:
    images, measured = test_data(subject)
    model = berg.get_encoding_model(MODEL_ID, subject=subject)
    preds = berg.encode(model, images)
    model.cleanup()
    dataset, sub = subject.rsplit("-", 1)
    pred = preds[dataset][f"sub-{sub}"]
    assert pred.shape == measured.shape, (subject, pred.shape, measured.shape)

    corr = pearsonr(pred, measured, axis=0)[0].astype(np.float32)
    median = float(np.median(corr))
    print(f"{subject:<9} test images {len(images):>5}  voxels {len(corr):>6}  median r {median:.4f}")
    correlations[subject], medians[subject] = corr, median
    table.append([subject, len(images), len(corr), f"{median:.4f}"])
    plot_histogram(corr, subject, median, os.path.join(OUT_DIR, f"correlation_histogram_{subject}.png"))

np.savez(CORRELATIONS_PATH, **correlations)
print(f"Saved {CORRELATIONS_PATH}")
with open(os.path.join(OUT_DIR, "median_correlation_per_subject.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "test_images", "voxels", "median_correlation"])
    writer.writerows(table)
plot_medians(medians, os.path.join(OUT_DIR, "median_correlation_per_subject.png"))
print(f"Saved in {OUT_DIR}")
