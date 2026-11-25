"""Correlation brain maps of the Universal Brain Encoder (NSD + LAION-fMRI), for BERG's encoding_models_accuracy/ folder.

Reads the per-voxel correlations saved by accuracy_universal_brain_encoder_nsd_laion.py (correlations.npz, in the
folder the script is run from) and the voxel positions and ROI labels from the model's metadata files. Per subject:
  - correlation_brain_map_<subject>.png: pycortex flat map of fsaverage, as BERG's NSD fsaverage maps (huze)
  - correlation_brain_map_rois_<subject>.png: flat map with ROI outlines, as the v3 plots (flatmap_plot.plot_voxel_heatmap),
    with the voxels and ROI masks from the metadata files

Needs pycortex installed. The pycortex fsaverage subject and the ROI plotting code are read from
PYCORTEX_FILESTORE and ROI_CODE_DIR.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import cortex
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BERG_DIR = "/home/romanb/data/BERG/berg_dir"
MODEL_DIR = "encoding_models/modality-fmri/train_dataset-nsd_laion_fmri_fsaverage/model-universal_brain_encoder"
ACCURACY_DIR = os.path.join(BERG_DIR, MODEL_DIR, "encoding_models_accuracy")  # maps out
CORRELATIONS_PATH = os.path.join(os.getcwd(), "correlations.npz")  # written by the accuracy script
METADATA_FILES = ["metadata_nsd.npy", "metadata_laion.npy"]  # in BERG_DIR/MODEL_DIR/metadata/

PYCORTEX_FILESTORE = "/home/romanb/data/miniconda3/envs/pycortex/share/pycortex/db"  # has the fsaverage subject
cortex.db.filestore = PYCORTEX_FILESTORE
cortex.db.reload_subjects()

ROI_CODE_DIR = "/home/romanb/data/FmriVision_re/experiments/blip/ROI/"  # flatmap_plot.py and the modules it imports
sys.path.append(ROI_CODE_DIR)
import flatmap_plot
import render_roi_panel
from render_roi_panel import _render_true_flatmap

CMAP = "hot"  # colors of both maps, 0 to 1
# the ROI plotting code builds its own colormap ("WhYlOrRd") inside its functions; make it CMAP instead
flatmap_plot._OVERLAY_CMAP = plt.get_cmap(CMAP)  # colorbar of the cropped (NSD) maps


class _MapColors:
    # replaces LinearSegmentedColormap in render_roi_panel, where it is used only to build the map's colormap
    @staticmethod
    def from_list(*args, **kwargs):
        return plt.get_cmap(CMAP)


render_roi_panel.LinearSegmentedColormap = _MapColors

NUM_VERTICES = 163842  # fsaverage7 vertices per hemisphere
# map with ROIs: settings of the v3 plots, colors 0 to 1 (NSD maps are also cropped as in v3, LAION maps are not)
ROI_MAP = dict(cbar_vmin=0.0, cbar_vmax=1.0, roi_label_fontsize=16, cbar_fontsize=16,
               roi_outline_linestyle="dotted", show_cbar=True)
NSD_CROP_TOP = 270
# ROIs outlined on the LAION maps: the NSD ones (flatmap_plot.ALL_ROIS) that LAION has; LAION has one FBA (no FBA-1/2)
LAION_MAP_ROIS = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA", "OFA", "FFA-1", "FFA-2",
                  "OPA", "PPA", "VWFA-1", "VWFA-2"]

# ── Correlations and metadata ─────────────────────────────────────────────────
CORRELATIONS = np.load(CORRELATIONS_PATH)
METADATA = {}  # {subject: entry}: vertex_inds, hemis, rois, ... of each voxel, in the model's voxel order
for fname in METADATA_FILES:
    METADATA.update(np.load(os.path.join(BERG_DIR, MODEL_DIR, "metadata", fname), allow_pickle=True).item())


# ── Plots ─────────────────────────────────────────────────────────────────────
def full_fsaverage(corr, meta):
    # values on all fsaverage7 vertices (left hemisphere, then right); NaN outside the subject's voxels
    values = np.full(2 * NUM_VERTICES, np.nan, dtype=np.float32)
    hemis = np.asarray(meta["hemis"], dtype=bool)
    values[np.asarray(meta["vertex_inds"]) + np.where(hemis, 0, NUM_VERTICES)] = corr
    return values


def plot_brain_map(values, subject, median, path):
    # pycortex flat map of fsaverage, as BERG's NSD fsaverage maps (huze), without ROIs
    vertex_data = cortex.Vertex(values, "fsaverage", cmap=CMAP, vmin=0, vmax=1, with_colorbar=True)
    fig = cortex.quickshow(vertex_data, with_curvature=True, curvature_brightness=0.5,
                           with_rois=False, with_labels=False, with_colorbar=True)
    plt.title(f"{subject} | Pearson r | median {median:.3f}", fontsize=30)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_voxel_heatmap_metadata(score_arr, meta, roi_names=flatmap_plot.ALL_ROIS, figsize=(24, 16),
                                roi_outline_style="alpha_shape", roi_outline_shrink=0.97, roi_outline_alpha=0.04,
                                cbar_vmin=None, cbar_vmax=None, roi_label_fontsize=14, cbar_fontsize=14,
                                roi_outline_linestyle="dashed", crop_top=None, cbar_n_ticks=5, show_cbar=True):
    # flatmap_plot.plot_voxel_heatmap ("all_vertices" mode), with the voxels and ROI masks taken from our
    # metadata file instead of the NSD mask files
    score_arr = np.asarray(score_arr, dtype=np.float32)
    hemis = np.asarray(meta["hemis"], dtype=bool)
    vertex_inds = np.asarray(meta["vertex_inds"])
    roi_inds, roi_labels = np.asarray(meta["rois"]["inds"]), np.asarray(meta["rois"]["labels"])
    roi_names = [r for r in roi_names if r in set(roi_labels)]  # the ROIs this subject has

    hemi_payload = {}
    for hemi, side, offset in [("left", hemis, 0), ("right", ~hemis, NUM_VERTICES)]:
        union_mask = np.zeros(NUM_VERTICES, dtype=np.int32)
        union_mask[vertex_inds[side]] = 1
        stat = np.full(NUM_VERTICES, np.nan, dtype=np.float32)
        stat[union_mask > 0] = score_arr[offset:offset + NUM_VERTICES][union_mask > 0]
        hemi_payload[hemi] = {"union_mask": union_mask, "combined_stat": stat,
                              "label_map": np.zeros(NUM_VERTICES, dtype=np.int32), "entries": []}
        for idx, roi in enumerate(roi_names):
            voxels = roi_inds[roi_labels == roi]
            mask = np.zeros(NUM_VERTICES, dtype=np.int32)
            mask[vertex_inds[voxels[side[voxels]]]] = 1
            hemi_payload[hemi]["label_map"][mask > 0] = idx + 1
            hemi_payload[hemi]["entries"].append({"roi": roi, "mask": mask})

    # from here on, as in flatmap_plot.plot_voxel_heatmap
    finite_vals = score_arr[np.isfinite(score_arr)]
    actual_vmin = float(np.percentile(finite_vals, 5)) if cbar_vmin is None else cbar_vmin
    actual_vmax = float(np.percentile(finite_vals, 95)) if cbar_vmax is None else cbar_vmax
    img = _render_true_flatmap(
        hemi_payload["left"], hemi_payload["right"], figsize,
        roi_outline_style=roi_outline_style, roi_outline_shrink=roi_outline_shrink,
        roi_outline_alpha=roi_outline_alpha, cbar_vmin=actual_vmin, cbar_vmax=actual_vmax,
        roi_label_fontsize=roi_label_fontsize, cbar_fontsize=cbar_fontsize,
        roi_outline_linestyle=roi_outline_linestyle, show_colorbar=show_cbar and crop_top is None,
    )
    if crop_top is not None:
        img = img.crop((0, crop_top, img.width, img.height))
        if show_cbar:
            pad_w = int(img.width * 0.12)
            cbar = flatmap_plot._make_vertical_colorbar(pad_w, img.height, actual_vmin, actual_vmax, cbar_fontsize,
                                                        n_ticks=cbar_n_ticks)
            combined = Image.new("RGB", (img.width + pad_w, img.height), "white")
            combined.paste(img, (0, 0))
            combined.paste(cbar, (img.width, max(0, (img.height - cbar.height) // 2)))
            img = combined
    return img


def add_title(img, title, banner_height=50, bg_color="white", text_color="black", font_size=20, line_color="black",
              line_width=2, shift_text=60):
    # title banner, as in the v3 plots
    img_with_title = Image.new("RGB", (img.width, img.height + banner_height), color=bg_color)
    img_with_title.paste(img, (0, banner_height))
    draw = ImageDraw.Draw(img_with_title)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=font_size)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.width - text_width) / 2 - shift_text
    y = (banner_height - text_height) / 2
    draw.text((x, y), title, fill=text_color, font=font)
    line_y = y + text_height + 4
    draw.line([(x, line_y), (x + text_width, line_y)], fill=line_color, width=line_width)
    return img_with_title


def plot_roi_map(values, subject, median, path):
    # flat map with ROI outlines, as the v3 plots; voxels and ROI masks from our metadata, for both datasets
    if subject.startswith("NSD-"):
        img = plot_voxel_heatmap_metadata(values, METADATA[subject], roi_names=flatmap_plot.ALL_ROIS,
                                          crop_top=NSD_CROP_TOP, **ROI_MAP)
    else:
        img = plot_voxel_heatmap_metadata(values, METADATA[subject], roi_names=LAION_MAP_ROIS, **ROI_MAP)
    add_title(img, f"{subject} | Pearson r | median {median:.3f}", font_size=60, shift_text=60).save(path)


# ── Maps per subject ──────────────────────────────────────────────────────────
for subject in CORRELATIONS.files:
    corr, meta = CORRELATIONS[subject], METADATA[subject]
    assert len(corr) == meta["num_voxels"], (subject, len(corr), meta["num_voxels"])
    median = float(np.median(corr))
    values = full_fsaverage(corr, meta)
    plot_brain_map(values, subject, median, os.path.join(ACCURACY_DIR, f"correlation_brain_map_{subject}.png"))
    plot_roi_map(values, subject, median, os.path.join(ACCURACY_DIR, f"correlation_brain_map_rois_{subject}.png"))
    print(f"{subject}: maps saved")
print(f"Saved in {ACCURACY_DIR}")
