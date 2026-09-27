"""Convert SMPL-order joint positions into the 41-column joint-angle array.

Usage (HumanML3D):
    python preprocess/compute_joint_angles.py --src data/HumanML3D/new_joints --dst data/HumanML3D/joint_angle
Usage (KIT-ML, after preprocess/kit_to_smpl_order.py):
    python preprocess/compute_joint_angles.py --src data/KIT-ML/new_joints_smpl_order --dst data/KIT-ML/joint_angle

Each output file has shape (T, 41). The 29 columns used by the model are
selected at load time (datasets/joint_angle_dataset.py: ALL_USED_INDICES).
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from angle_utils import *  # noqa: F401,F403


def joint_angles_from_positions(data_joints: np.ndarray) -> np.ndarray:
    """data_joints: (T, J>=22, 3) joint positions in SMPL joint order."""
    time_steps = data_joints.shape[0]
    ric_data = data_joints[:, :, [2, 1, 0]]  # Reorder to (z, y, x)
    ric_data[:, :, 2] *= -1
    frame = ric_data.shape[0]
    pelvis = ric_data[:,0,:]
    right_hip = ric_data[:,2,:]
    right_knee = ric_data[:,5,:]
    right_ankle = ric_data[:,8,:]
    right_foot = ric_data[:,11,:]
    right_shoulder = ric_data[:,17,:]
    right_elbow = ric_data[:, 19, :]
    right_wrist = ric_data[:, 21, :]
    
    left_hip = ric_data[:,1,:]
    left_knee = ric_data[:,4,:]
    left_ankle = ric_data[:,7,:]
    left_foot = ric_data[:,10,:]
    left_shoulder = ric_data[:, 16, :]
    left_elbow = ric_data[:, 18, :]
    left_wrist = ric_data[:, 20, :]
    
    neck = ric_data[:,12,:]
    head = ric_data[:,15,:]
    
    body_coordinate = compute_body_coordinate(right_hip, left_hip,right_shoulder,left_shoulder,pelvis, smooth_forward=False)
    
    i = 0
    for coordinate in body_coordinate:
        if not is_right_hand_coordinate(coordinate):
            print("wrong body_coordinate calculation index:", i)
        i+=1
    right_hand_axes = np.array([
        [1, 0, 0],  # X-axis
        [0, 1, 0],  # Y-axis
        [0, 0, 1]  # Z-axis
    ])
    right_hand_axes_with_nframe = np.tile(right_hand_axes, (ric_data.shape[0], 1, 1))
    pelvis_angles = calculate_pelvis_euler_angles(body_coordinate)
    
    right_hip_angles = calculate_hip_angles(right_knee-right_hip, body_coordinate)
    right_hip_angles[:, 1] = compute_signed_angle_xy_projection(right_knee-right_hip, body_coordinate)
    right_hip_angles[:, 2] = compute_hip_rotation(right_hip_angles[:, :2], right_hip, right_knee, right_ankle, body_coordinate)
    
    left_hip_angles = calculate_hip_angles(left_knee-left_hip, body_coordinate)
    left_hip_angles[:, 1] = compute_signed_angle_xy_projection(left_knee-left_hip, body_coordinate)
    left_hip_angles[:, 2] = compute_hip_rotation(left_hip_angles[:, :2], left_hip, left_knee, left_ankle, body_coordinate)
    
    right_knee_angle = calculate_knee_angle(right_hip, right_knee, right_ankle)
    left_knee_angle = calculate_knee_angle(left_hip, left_knee, left_ankle)
    right_foot_angle = calculate_ankle_angle(right_knee, right_ankle, right_foot)
    left_foot_angle = calculate_ankle_angle(left_knee, left_ankle, left_foot)
    
    lumbar_vector = neck - pelvis  # shape: (n_frames, 3)
    
    lumbar_extension = calculate_lumbar_angles(lumbar_vector, body_coordinate)[:, 0]  # Only flexion
    
    lumbar_bending = compute_signed_angle_xy_projection(lumbar_vector, body_coordinate)
    
    lumbar_angles = np.stack([lumbar_extension, lumbar_bending], axis=1)
    lumbar_rotation, neck_coordinate = compute_lumbar_rotation(lumbar_angles, neck, right_shoulder, body_coordinate)
    
    shoulder_coordinate = compute_shoulder_coordinate(body_coordinate, lumbar_extension, lumbar_bending, lumbar_rotation, neck_coordinate)
    
    right_shoulder_angles = calculate_shoulder_angles(right_elbow-right_shoulder, shoulder_coordinate)
    right_shoulder_angles[:, 1] = compute_signed_angle_xy_projection(right_elbow-right_shoulder, shoulder_coordinate)
    right_shoulder_angles[:, 2] = compute_shoulder_rotation(right_shoulder_angles[:, :2], right_elbow, right_wrist, shoulder_coordinate)
    
    left_shoulder_angles = calculate_shoulder_angles(left_elbow-left_shoulder, shoulder_coordinate)
    left_shoulder_angles[:, 1] = compute_signed_angle_xy_projection(left_elbow-left_shoulder, shoulder_coordinate)
    left_shoulder_angles[:, 2] = compute_shoulder_rotation(left_shoulder_angles[:, :2], left_elbow, left_wrist, shoulder_coordinate)
    
    right_elbow_angle = calculate_elbow_angle(right_shoulder, right_elbow, right_wrist)
    left_elbow_angle = calculate_elbow_angle(left_shoulder, left_elbow, left_wrist)
    
    neck_angles = calculate_neck_angles(head - neck, shoulder_coordinate)
    neck_angles[:, 1] = compute_signed_angle_xy_projection(head - neck, shoulder_coordinate)
    
    
    joint_data = {
        "pelvis_tilt": pelvis_angles[:,0],
        "pelvis_list": pelvis_angles[:,1],
        "pelvis_rotation": pelvis_angles[:,2],
        "pelvis_tx": pelvis[:,0],
        "pelvis_ty": pelvis[:,1],
        "pelvis_tz": pelvis[:,2],
        "right_hip_tx": right_hip[:, 0],
        "right_hip_ty": right_hip[:, 1],
        "right_hip_tz": right_hip[:, 2],
        "left_hip_tx": left_hip[:, 0],
        "left_hip_ty": left_hip[:, 1],
        "left_hip_tz": left_hip[:, 2],
        "right_shoulder_tx": right_shoulder[:, 0],
        "right_shoulder_ty": right_shoulder[:, 1],
        "right_shoulder_tz": right_shoulder[:, 2],
        "left_shoulder_tx": left_shoulder[:, 0],
        "left_shoulder_ty": left_shoulder[:, 1],
        "left_shoulder_tz": left_shoulder[:, 2],
        "hip_flexion_r": right_hip_angles[:,0],
        "hip_adduction_r": -right_hip_angles[:,1],
        "hip_rotation_r": right_hip_angles[:,2],
        "knee_angle_r": right_knee_angle,
        "ankle_angle_r": right_foot_angle - 90,
        "hip_flexion_l": left_hip_angles[:,0],
        "hip_adduction_l": left_hip_angles[:,1],
        "hip_rotation_l": -left_hip_angles[:,2],   # for opensim it should have a - sign
        "knee_angle_l": left_knee_angle,
        "ankle_angle_l": left_foot_angle - 90,
        "lumbar_extension": -lumbar_extension,   # because here is extension, not flexion
        "lumbar_bending": lumbar_bending,
        "lumbar_rotation": lumbar_rotation,
        "arm_flex_r": right_shoulder_angles[:,0],
        "arm_add_r": -right_shoulder_angles[:,1],
        "arm_rot_r": -right_shoulder_angles[:,2],
        "arm_flex_l": left_shoulder_angles[:, 0],
        "arm_add_l": left_shoulder_angles[:, 1],
        "arm_rot_l": -left_shoulder_angles[:, 2],
        "elbow_flex_r": right_elbow_angle,
        "elbow_flex_l": left_elbow_angle,
        "neck_flexion": neck_angles[:,0],
        "neck_adduction": neck_angles[:,1],
    }
    return np.column_stack([np.asarray(v) for v in joint_data.values()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of (T, J, 3) joint-position .npy files")
    ap.add_argument("--dst", required=True, help="output directory for (T, 41) joint-angle .npy files")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    files = sorted(f for f in os.listdir(args.src) if f.endswith(".npy"))
    bad = []
    for f in tqdm(files):
        joints = np.load(os.path.join(args.src, f))
        if joints.ndim != 3:
            bad.append(f)
            continue
        np.save(os.path.join(args.dst, f), joint_angles_from_positions(joints))
    if bad:
        print(f"Skipped {len(bad)} files with unexpected shape: {bad[:10]}")


if __name__ == "__main__":
    main()
