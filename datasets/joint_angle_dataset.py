# Modified from MoPatch (https://github.com/line/MotionPatches),
# Copyright 2024 LY Corporation, licensed under CC BY-NC 4.0
# (https://creativecommons.org/licenses/by-nc/4.0/).

import random
from os.path import join as pjoin
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from ._stores import open_motion_store, open_text_store

# Fine-Grained Body Part Representation: 14 distinct body parts
# The MotionEncoder in clip.py relies on this exact order for splitting.
# Total: 3+3+3+3+1+1+1+1+3+2+3+3+1+1 = 29 dimensions
# Each part will be projected to 16-dim embedding, resulting in 14*16=224 feature width
ALL_USED_INDICES = [
    # 1. Global Pos (3 dims): indices [3, 4, 5]
    3, 4, 5,
    # 2. Pelvis (3 dims): indices [0, 1, 2]
    0, 1, 2,
    # 3. Right Hip (3 dims): indices [18, 19, 20]
    18, 19, 20,
    # 4. Left Hip (3 dims): indices [23, 24, 25]
    23, 24, 25,
    # 5. Right Knee (1 dim): index [21]
    21,
    # 6. Left Knee (1 dim): index [26]
    26,
    # 7. Right Ankle (1 dim): index [22]
    22,
    # 8. Left Ankle (1 dim): index [27]
    27,
    # 9. Lumbar (3 dims): indices [28, 29, 30]
    28, 29, 30,
    # 10. Neck (2 dims): indices [39, 40]
    39, 40,
    # 11. Right Shoulder (3 dims): indices [31, 32, 33]
    31, 32, 33,
    # 12. Left Shoulder (3 dims): indices [34, 35, 36]
    34, 35, 36,
    # 13. Right Elbow (1 dim): index [37]
    37,
    # 14. Left Elbow (1 dim): index [38]
    38,
]

class JointAngleMotionDataset(data.Dataset):
    def __init__(
        self,
        cfg,
        mean: np.ndarray,
        std: np.ndarray,
        split_file: str,
        eval_mode: bool = False,
        patch_size: int = 16,
        fps: Optional[bool] = None,
    ):
        self.cfg = cfg
        self.eval_mode = eval_mode
        self.max_motion_length = cfg.dataset.max_motion_length
        self.fps = fps
        self.mean = mean
        self.std = std

        # Load data logic
        data_dict = {}
        id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        motion_store = open_motion_store(cfg.dataset.joint_angle_dir)
        text_store = open_text_store(cfg.dataset.text_dir)

        print(f"Loading joint angle data from {cfg.dataset.joint_angle_dir}")
        for name in tqdm(id_list, desc="Loading joint angles"):
            try:
                motion = motion_store[name]
                if len(motion.shape) != 2 or motion.shape[1] not in [40, 41] or np.isnan(motion).any():
                    continue

                text_data = []
                flag = False
                entries = text_store[name]
                for index, entry in enumerate(entries):
                    if eval_mode and index >= 1: continue

                    text_dict = {"caption": entry["caption"], "tokens": entry["tokens"]}
                    f_tag = entry["f_tag"]
                    to_tag = entry["to_tag"]

                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        # Handling segments
                        fps_val = cfg.dataset.fps if hasattr(cfg.dataset, 'fps') else 20
                        n_motion = motion[int(f_tag * fps_val) : int(to_tag * fps_val)]
                        new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                        while new_name in data_dict:
                            new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                        data_dict[new_name] = {"motion": n_motion, "length": len(n_motion), "text": [text_dict]}

                if flag:
                    data_dict[name] = {"motion": motion, "length": len(motion), "text": text_data}

            except Exception:
                pass

        self.name_list = sorted(list(data_dict.keys()))
        self.data_dict = data_dict

        # Preprocessing
        print("Preprocessing joint angle motions...")
        for key in tqdm(data_dict.keys()):
            motion = data_dict[key]["motion"]
            # Normalize
            motion = (motion - self.mean[np.newaxis, :]) / (self.std[np.newaxis, :] + 1e-8)
            data_dict[key]["pre_motion"] = motion
            data_dict[key]["length"] = motion.shape[0]

    def _create_motion_patches(self, motion: np.ndarray) -> torch.Tensor:
        """Selects the 29 relevant joint angle columns."""
        motion_tensor = torch.from_numpy(motion).float()
        return motion_tensor[:, ALL_USED_INDICES]

    def __len__(self) -> int:
        return len(self.data_dict) * self.cfg.dataset.times

    def __getitem__(self, item: int) -> Tuple[str, torch.Tensor, int, int]:
        idx = item % len(self.data_dict)
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["pre_motion"], data["length"], data["text"]

        caption = text_list[0]["caption"] if self.eval_mode else random.choice(text_list)["caption"]
        # caption = text_list[0]["caption"] 

        # Padding / Cutting
        if m_length >= self.max_motion_length:
            idx_start = random.randint(0, m_length - self.max_motion_length) if not self.eval_mode else 0
            motion = motion[idx_start : idx_start + self.max_motion_length]
            m_length = self.max_motion_length
        else:
            if self.cfg.preprocess.padding:
                padding = np.zeros((self.max_motion_length - m_length, motion.shape[1]), dtype=np.float32)
                motion = np.concatenate((motion, padding), axis=0)

        # Output shape: (T, 29)
        motion_angles = self._create_motion_patches(motion)
        return caption, motion_angles, m_length, item


def compute_stats(data_dir: str, file_list: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global mean and standard deviation for joint angle data.

    Args:
        data_dir: Directory containing .npy files with joint angle data
        file_list: Optional list of specific files to process. If None, process all .npy files

    Returns:
        Tuple of (mean, std) where each has shape [D] (D is 40 or 41 depending on data)
    """
    import os

    all_data = []
    detected_dims = None

    store = open_motion_store(data_dir)
    if file_list is None:
        file_list = list(store.keys())

    print(f"Computing statistics over {len(file_list)} files...")

    for filename in tqdm(file_list, desc="Loading files"):
        try:
            data = store[filename]

            # Validate shape
            if len(data.shape) != 2:
                print(f"Skipping {filename}: invalid shape {data.shape}, expected (T, D)")
                continue

            # Auto-detect dimensions from first valid file
            if detected_dims is None:
                detected_dims = data.shape[1]
                print(f"Detected {detected_dims} dimensions from data")

            if data.shape[1] != detected_dims:
                print(f"Skipping {filename}: inconsistent dimensions {data.shape[1]}, expected {detected_dims}")
                continue

            if np.isnan(data).any():
                print(f"Skipping {filename}: contains NaN")
                continue

            all_data.append(data)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    if len(all_data) == 0:
        raise ValueError("No valid data files found!")

    # Concatenate all data along time dimension
    all_data = np.concatenate(all_data, axis=0)  # Shape: (total_frames, D)

    print(f"Total frames: {all_data.shape[0]}")
    print(f"Dimensions: {all_data.shape[1]}")

    # Compute statistics
    mean = all_data.mean(axis=0)  # Shape: [D]
    std = all_data.std(axis=0)    # Shape: [D]

    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)

    print(f"Mean shape: {mean.shape}, range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Std shape: {std.shape}, range: [{std.min():.4f}, {std.max():.4f}]")

    return mean, std
