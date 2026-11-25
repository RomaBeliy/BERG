"""
Build the NSD metadata file for BERG, in the same format as the LAION-fMRI one.

Output: one .npy file holding a dict with one entry per subject:
    {"NSD-01": {
        "num_voxels":  int,
        "vertex_inds": int64 [V],       fsaverage7 vertex index within its hemisphere
        "hemis":       bool [V],        True = left hemisphere
        "coords":      float32 [V, 3],  fsaverage7 SURFACE coordinates in mm
        "rois":        {"inds": int64 [K], "labels": list of K str},  one entry per (voxel, ROI)
        "roi_groups":  {"inds": int64 [K], "labels": list of K str},  one entry per (voxel, group)
     }, ...}
V = number of voxels, in the model's output order (left hemisphere, then right).
"inds" are voxel indices; a voxel appears once per ROI/group, so overlaps are kept.
ROI labels like "faces_FFA-1" are split into ROI "FFA-1" (in "rois") and group "faces" (in "roi_groups").
Groups "fplaces" and "visual" are renamed "places" and "retinotopic"; "streams" is not stored as a group.

Load with:
    metadata = np.load("metadata_nsd.npy", allow_pickle=True).item()
"""
import os

import numpy as np

# read only: nothing is written to these folders
MASK_DIR = "/home/romanb/data/datasets/NVD/rois/subj{s:02d}"
NUM_VOXELS_PATH = "/home/romanb/data/datasets/NVD/tutorial_data/subjects/subsuper/num_voxels_all_subjects.npy"
ROI_PATH = "/home/romanb/data/datasets/NVD/rois/roi_label_all_subs.npz"
GEOMETRY_PATH = ("/net/mraid20/ifs/wisdom/groups/iranig/datasets/laion_fmri/exports/"
                 "encoder_spatial_metadata_v1/fsaverage7_geometry.npz")
# output is written locally, in the folder you run the script from
OUT_PATH = os.path.join(os.getcwd(), "metadata_nsd.npy")

SUBJECTS = range(1, 9)
SURFACE = "pial"  # "pial" or "white", same as in the LAION metadata

# naming shared with the LAION metadata: NSD ROI names ("FFA-1", "mfs-words"),
# groups "places" and "retinotopic" (LAION's names) instead of "fplaces" and "visual"
GROUP_RENAME = {"fplaces": "places", "visual": "retinotopic"}
NO_GROUP = {"streams"}  # its ROIs (early, ventral, ...) are kept, but "streams" itself is not a group

out_path = os.path.realpath(OUT_PATH)
for src in [os.path.dirname(os.path.dirname(MASK_DIR)), os.path.dirname(NUM_VOXELS_PATH),
            os.path.dirname(ROI_PATH), os.path.dirname(GEOMETRY_PATH)]:
    src = os.path.realpath(src)
    assert os.path.commonpath([src, out_path]) != src, \
        f"Refusing to write inside a source folder: {out_path}. Run the script from a local folder."

geometry = np.load(GEOMETRY_PATH)
surface = {"lh": geometry[f"{SURFACE}_L_xyz_mm"], "rh": geometry[f"{SURFACE}_R_xyz_mm"]}

# voxel counts per subject, and where each subject starts in the global voxel order
num_voxels_subjects = np.load(NUM_VOXELS_PATH).sum(1).astype(int)
start_ind = np.cumsum(num_voxels_subjects) - num_voxels_subjects

# ROI labels: global voxel indices over all subjects, labels like "faces_FFA-1"
roi_data = np.load(ROI_PATH)
roi_inds = roi_data["inds"].astype(np.int64)
roi_labels = [l.decode() if isinstance(l, bytes) else str(l) for l in roi_data["labels"]]
assert all("_" in l for l in roi_labels), "expected labels like 'faces_FFA-1'"
roi_groups_all = np.array([GROUP_RENAME.get(l.split("_", 1)[0], l.split("_", 1)[0]) for l in roi_labels])
roi_names_all = np.array([l.split("_", 1)[1] for l in roi_labels])

metadata = {}
for i, s in enumerate(SUBJECTS):
    subject = f"NSD-{s:02d}"
    lh_mask = np.load(os.path.join(MASK_DIR.format(s=s), "lh.all-vertices_fsaverage_space.npy")).astype(bool)
    rh_mask = np.load(os.path.join(MASK_DIR.format(s=s), "rh.all-vertices_fsaverage_space.npy")).astype(bool)
    lh_inds, rh_inds = np.flatnonzero(lh_mask), np.flatnonzero(rh_mask)

    num_voxels = len(lh_inds) + len(rh_inds)
    assert num_voxels == num_voxels_subjects[i], (subject, num_voxels, num_voxels_subjects[i])

    vertex_inds = np.concatenate([lh_inds, rh_inds]).astype(np.int64)
    hemis = np.concatenate([np.ones(len(lh_inds), bool), np.zeros(len(rh_inds), bool)])
    coords = np.concatenate([surface["lh"][lh_inds], surface["rh"][rh_inds]]).astype(np.float32)

    # this subject's ROI entries, converted to subject voxel indices
    sel = (roi_inds >= start_ind[i]) & (roi_inds < start_ind[i] + num_voxels)
    inds = roi_inds[sel] - start_ind[i]
    # one entry per (voxel, group), even if the voxel is in several ROIs of that group
    group_pairs = sorted(set((v, g) for v, g in zip(inds.tolist(), roi_groups_all[sel].tolist())
                             if g not in NO_GROUP))

    metadata[subject] = {
        "num_voxels": num_voxels,
        "vertex_inds": vertex_inds,
        "hemis": hemis,
        "coords": coords,
        "rois": {"inds": inds, "labels": roi_names_all[sel].tolist()},
        "roi_groups": {"inds": np.array([p[0] for p in group_pairs], dtype=np.int64),
                       "labels": [p[1] for p in group_pairs]},
    }
    print(f"{subject}: {num_voxels} voxels ({len(lh_inds)} lh, {len(rh_inds)} rh)")
    print(f"  ROIs:   {sorted(set(metadata[subject]['rois']['labels']))}")
    print(f"  groups: {sorted(set(metadata[subject]['roi_groups']['labels']))}")

np.save(out_path, metadata, allow_pickle=True)
print("saved", out_path)
