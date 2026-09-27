"""Compute the per-column normalization statistics (Mean_angle.npy / Std_angle.npy).

NOTE: the released checkpoints were trained with the statistics shipped in
data/HumanML3D and data/KIT-ML of this repository. Use those files to evaluate the
released checkpoints; this script is only needed when training on new data.
(For HumanML3D, running it over all.txt reproduces the shipped file exactly.)

Usage:
    python preprocess/compute_stats.py --data_root data/HumanML3D --split all.txt
"""
import argparse
import os
import sys
from os.path import join as pjoin

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import compute_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="all.txt")
    ap.add_argument("--angle_dir", default="joint_angle")
    args = ap.parse_args()
    names = [l.strip() for l in open(pjoin(args.data_root, args.split)) if l.strip()]
    mean, std = compute_stats(pjoin(args.data_root, args.angle_dir), names)
    np.save(pjoin(args.data_root, "Mean_angle.npy"), mean)
    np.save(pjoin(args.data_root, "Std_angle.npy"), std)
    print("saved Mean_angle.npy / Std_angle.npy to", args.data_root)


if __name__ == "__main__":
    main()
