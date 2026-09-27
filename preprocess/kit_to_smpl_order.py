"""Re-order KIT-ML joints (21) into the SMPL joint order (24) expected by
preprocess/compute_joint_angles.py. SMPL joints absent from KIT-ML are set to zero
(they are not used by the joint-angle computation).

Usage:
    python preprocess/kit_to_smpl_order.py --src data/KIT-ML/new_joints --dst data/KIT-ML/new_joints_smpl_order
"""
import argparse
import os

import numpy as np

KIT_JOINT_NAMES = [
    "fake_pelvis", "pelvis", "neck", "upper_neck", "head",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle", "right_ankle_bad", "right_foot",
    "left_hip", "left_knee", "left_ankle", "left_ankle_bad", "left_foot",
]
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]
# SMPL index -> KIT index (None: joint not present in KIT-ML)
MAPPING = {i: (KIT_JOINT_NAMES.index(n) if n in KIT_JOINT_NAMES else None)
           for i, n in enumerate(SMPL_JOINT_NAMES)}


def kit_to_smpl(kit_data: np.ndarray) -> np.ndarray:
    """(T, 21, 3) KIT-ML joints -> (T, 24, 3) in SMPL order."""
    out = np.zeros((kit_data.shape[0], len(SMPL_JOINT_NAMES), 3), dtype=kit_data.dtype)
    for smpl_idx, kit_idx in MAPPING.items():
        if kit_idx is not None:
            out[:, smpl_idx] = kit_data[:, kit_idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    for f in sorted(os.listdir(args.src)):
        if not f.endswith(".npy"):
            continue
        kit = np.load(os.path.join(args.src, f))
        if kit.ndim != 3 or kit.shape[1:] != (21, 3):
            print(f"Skipping {f}: unexpected shape {kit.shape}")
            continue
        np.save(os.path.join(args.dst, f), kit_to_smpl(kit))


if __name__ == "__main__":
    main()
