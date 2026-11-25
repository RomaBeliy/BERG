"""
Build the LAION-fMRI metadata file for BERG from the LAION encoder spatial-metadata export.

Output: one .npy file holding a dict with one entry per subject:
    {"LAION-01": {
        "num_voxels":  int,
        "vertex_inds": int64 [V],       fsaverage7 vertex index within its hemisphere
        "hemis":       bool [V],        True = left hemisphere
        "coords":      float32 [V, 3],  fsaverage7 SURFACE coordinates in mm
        "rois":        {"inds": int64 [K], "labels": list of K str},  one entry per (voxel, ROI)
        "roi_groups":  {"inds": int64 [K], "labels": list of K str},  one entry per (voxel, group)
     }, ...}
V = number of response columns (voxels), in the model's output order.
"inds" are response-column indices; a voxel appears once per ROI/group, so overlaps are kept.
ROI names use the NSD naming ("FFA1" -> "FFA-1", "mfswords" -> "mfs-words"); groups keep LAION's names.

Load with:
    metadata = np.load("metadata_laion.npy", allow_pickle=True).item()
"""
import os
import re

import numpy as np

# read only: nothing is written to this folder
EXPORT_DIR = "/net/mraid20/ifs/wisdom/groups/iranig/datasets/laion_fmri/exports/encoder_spatial_metadata_v1"
# output is written locally, in the folder you run the script from
OUT_PATH = os.path.join(os.getcwd(), "metadata_laion.npy")
SUBJECTS = ["01", "03", "05", "06", "07"]
SURFACE = "pial"  # "pial" or "white", same as in the NSD metadata

N_VERTS = 163842  # fsaverage7 vertices per hemisphere
EXPECTED_VOXELS = {"01": 41575, "03": 37744, "05": 36798, "06": 35842, "07": 41892}  # from the export README

# naming shared with the NSD metadata: the NSD ROI names
NSD_ROIS = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies",
            "OFA", "FFA-1", "FFA-2", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words",
            "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]


def to_nsd_name(name):
    # e.g. FFA1 -> FFA-1, VWFA2 -> VWFA-2, mfswords -> mfs-words; other names are kept
    name = re.sub(r"^(FFA|FBA|VWFA)(\d)$", r"\1-\2", name)
    return {"mfswords": "mfs-words", "mTLwords": "mTL-words", "mTLbodies": "mTL-bodies",
            "mTLfaces": "mTL-faces", "aTLfaces": "aTL-faces"}.get(name, name)


def to_pairs(npz, rename=lambda name: name):
    # {name: column indices} -> paired lists, one entry per (voxel, name)
    inds, labels = [], []
    for name in npz.files:
        idx = npz[name].astype(np.int64).ravel()
        inds.append(idx)
        labels += [rename(name)] * len(idx)
    inds = np.concatenate(inds) if inds else np.zeros(0, dtype=np.int64)
    return {"inds": inds, "labels": labels}


export_dir = os.path.realpath(EXPORT_DIR)
out_path = os.path.realpath(OUT_PATH)
assert os.path.commonpath([export_dir, out_path]) != export_dir, \
    f"Refusing to write inside the export folder: {out_path}. Run the script from a local folder."

geometry = np.load(os.path.join(EXPORT_DIR, "fsaverage7_geometry.npz"))
surface = {"lh": geometry[f"{SURFACE}_L_xyz_mm"], "rh": geometry[f"{SURFACE}_R_xyz_mm"]}

metadata = {}
for s in SUBJECTS:
    sub_dir = os.path.join(EXPORT_DIR, f"sub-{s}")

    # response column i -> bilateral vertex index (left hemisphere first, then right)
    mask = np.load(os.path.join(sub_dir, "visual_mask_fsaverage7_full.npy")).astype(bool)
    assert mask.shape == (2 * N_VERTS,), mask.shape
    bilateral = np.flatnonzero(mask)
    num_voxels = len(bilateral)
    assert num_voxels == EXPECTED_VOXELS[s], (s, num_voxels)

    hemis = bilateral < N_VERTS
    vertex_inds = np.where(hemis, bilateral, bilateral - N_VERTS).astype(np.int64)

    coords = np.zeros((num_voxels, 3), dtype=np.float32)
    coords[hemis] = surface["lh"][vertex_inds[hemis]]
    coords[~hemis] = surface["rh"][vertex_inds[~hemis]]

    rois = to_pairs(np.load(os.path.join(sub_dir, "roi_indices.npz")), rename=to_nsd_name)
    roi_groups = to_pairs(np.load(os.path.join(sub_dir, "roi_group_indices.npz")))
    for r in (rois, roi_groups):
        assert r["inds"].size == 0 or (r["inds"].min() >= 0 and r["inds"].max() < num_voxels)

    metadata[f"LAION-{s}"] = {
        "num_voxels": num_voxels,
        "vertex_inds": vertex_inds,
        "hemis": hemis,
        "coords": coords,
        "rois": rois,
        "roi_groups": roi_groups,
    }
    print(f"LAION-{s}: {num_voxels} voxels ({hemis.sum()} lh, {(~hemis).sum()} rh)")
    print(f"  ROIs:   {sorted(set(rois['labels']))}")
    print(f"  groups: {sorted(set(roi_groups['labels']))}")
    print(f"  ROIs not in NSD: {sorted(set(rois['labels']) - set(NSD_ROIS))}")

np.save(out_path, metadata, allow_pickle=True)
print("saved", out_path)
