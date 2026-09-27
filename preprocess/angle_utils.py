"""Joint-angle kinematics used to build the Motion Image (paper Sec. 3.1, Supp. Sec. 1).

Angles follow the ISB joint coordinate conventions. All functions operate on
per-frame joint positions of shape (T, 3) or body frames of shape (T, 3, 3).
Only numpy/scipy are required.
"""
import numpy as np
from scipy import ndimage as filters
from scipy.spatial.transform import Rotation as R


def is_right_hand_coordinate(coordinate_axes):
    """
    Check whether the given coordinate system is a right-hand coordinate system.

    Args:
        coordinate_axes (np.ndarray): A (3,3) array where each row is a unit vector
                                      representing the X, Y, and Z axes.

    Returns:
        bool: True if the coordinate system is right-handed, False otherwise.
    """
    # Ensure coordinate axes are normalized
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=1, keepdims=True)

    # Check orthogonality using dot product
    dot_xy = np.dot(coordinate_axes[0], coordinate_axes[1])
    dot_yz = np.dot(coordinate_axes[1], coordinate_axes[2])
    dot_zx = np.dot(coordinate_axes[2], coordinate_axes[0])

    orthogonality = np.allclose([dot_xy, dot_yz, dot_zx], [0, 0, 0], atol=1e-6)

    # Check right-hand rule using cross product
    cross_product = np.cross(coordinate_axes[0], coordinate_axes[1])  # e_x × e_y
    right_hand = np.allclose(cross_product, coordinate_axes[2], atol=1e-3)
    # return orthogonality
    return orthogonality and right_hand

def compute_body_coordinate(right_hip, left_hip, right_shoulder, left_shoulder, pelvis, smooth_forward=False):
    across = right_hip - left_hip  # Hip width direction
    # across2 = right_shoulder - left_shoulder
    # across = across1 + across2
    across = across / np.linalg.norm(across, axis=-1, keepdims=True)  # Normalize across

    # Compute "up" direction using shoulder and hip positions
    # pelvis_mid = (right_hip + left_hip) / 2
    # shoulder_mid = (right_shoulder + left_shoulder) / 2
    hip_mid = (right_hip + left_hip) / 2
    pelvis_mid = pelvis
    up = pelvis_mid - hip_mid  # Define "up" properly
    # up = np.tile([0,1,0], (frame, 1))
    up = up / np.linalg.norm(up, axis=-1, keepdims=True)  # Normalize up

    # Make across truly perpendicular to up
    # across = np.cross(up, np.cross(across, up))
    # across = across / np.linalg.norm(across, axis=-1, keepdims=True)
    up = up - np.sum(up * across, axis=-1, keepdims=True) * across
    # Compute forward as the cross product of across and up
    forward = np.cross(up, across)
    forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)  # Normalize forward

    if smooth_forward:
        forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')

    human_coordinate = np.array([forward, up, across])
    combined_coordinates = np.transpose(human_coordinate, (1, 0, 2))
    return combined_coordinates

# def calculate_pelvis_angle(body_coordinate_vectors):
#     """
#     Calculate the signed angles between vectors and the three coordinate planes (XY, YZ, XZ).
#
#     Args:
#         vectors (np.ndarray): Shape (n_frame, 3), the vectors whose angles are computed.
#         coordinate_axes (np.ndarray): Shape (n_frame, 3, 3), custom coordinate system per frame.
#
#     Returns:
#         np.ndarray: Signed angles (n_frame, 3), in degrees, where columns represent angles to XY, YZ, and XZ planes.
#         :param body_coordinate_vectors:
#     """
#     forward = body_coordinate_vectors[:, 0, :]
#     up = body_coordinate_vectors[:, 1, :]
#     across = body_coordinate_vectors[:, 2, :]
#     n_frame = forward.shape[0]
#
#     angles_to_planes = np.zeros((n_frame, 3))  # Store (rotation, tilt, list) for each frame
#
#     for i in range(n_frame):
#         # Normalize the vectors
#         forward_frame = forward[i] / np.linalg.norm(forward[i])
#         up_frame = up[i] / np.linalg.norm(up[i])
#         across_frame = across[i] / np.linalg.norm(across[i])
#
#         # Angle with YZ plane (normal is [1, 0, 0]) pelvis_tilt
#         angles_to_planes[i, 0] = -np.degrees(np.arctan2(up_frame[0], np.linalg.norm(up_frame[1:])))
#         if up_frame[1] < 0 and forward_frame[0] < 0 and up_frame[0] > 0:  # Below XY plane, adjust range
#             angles_to_planes[i, 0] = -180 - angles_to_planes[i, 0]
#         elif up_frame[1] < 0 and forward_frame[0] < 0 and up_frame[0] < 0:
#             angles_to_planes[i, 0] = 180 - angles_to_planes[i, 0]
#         else:
#             angles_to_planes[i, 0] = angles_to_planes[i, 0]
#
#         # Angle with XY plane (normal is [0, 0, 1]) pelvis_list
#         angles_to_planes[i, 1] = np.degrees(np.arctan2(up_frame[2], np.linalg.norm(up_frame[:2])))
#         if up_frame[1] < 0 and across_frame[2] < 0 and up_frame[2] > 0:  # Below XY plane, adjust range
#             angles_to_planes[i, 1] = 180 - angles_to_planes[i, 1]
#         elif up_frame[1] < 0 and across_frame[2] < 0 and up_frame[2] < 0:
#             angles_to_planes[i, 1] = -180 - angles_to_planes[i, 1]
#         else:
#             angles_to_planes[i, 1] = angles_to_planes[i, 1]
#
#         # Angle with YZ plane (normal is [1, 0, 0]) pelvis_rotation
#         angles_to_planes[i, 2] = np.degrees(np.arctan2(across_frame[0], np.linalg.norm(across_frame[1:])))
#         if across_frame[2] < 0 and forward_frame[0] < 0 and across_frame[0] > 0:  # Below XY plane, adjust range
#             angles_to_planes[i, 2] = 180 - angles_to_planes[i, 2]
#         elif across_frame[2] < 0 and forward_frame[0] < 0 and across_frame[0] < 0:
#             angles_to_planes[i, 2] = -180 -  angles_to_planes[i, 2]
#         else:
#             angles_to_planes[i, 2] =  angles_to_planes[i, 2]
#
#     return angles_to_planes

def calculate_pelvis_euler_angles(body_coordinate_vectors):
    """
    Calculate the pelvis orientation (flexion/extension, adduction/abduction, rotation)
    using intrinsic ZXY Euler angles, from global coordinate to body coordinate.

    Args:
        body_coordinate_vectors (np.ndarray): (n_frame, 3, 3), body coordinate frame at each time point.

    Returns:
        np.ndarray: (n_frame, 3), Euler angles in degrees: [flexion (X), adduction (Y), rotation (Z)].
    """
    n_frame = body_coordinate_vectors.shape[0]
    euler_angles = np.zeros((n_frame, 3))

    for i in range(n_frame):
        # Body axes: each row is a body axis in global coordinates
        # Rows should be: [forward; up; across]
        # rotation_matrix = body_coordinate_vectors[i].T  # transpose to get global-to-body rotation
        rotation_matrix = body_coordinate_vectors[i].T
        # Create rotation object
        rot = R.from_matrix(rotation_matrix)

        # Extract intrinsic ZXY Euler angles (global -> body)
        # Intrinsic means rotations are applied in order around the moving axes
        angles = rot.as_euler('ZXY', degrees=True)
        recover = R.from_euler('ZXY', angles, degrees=True).as_matrix()

        euler_angles[i] = angles

    return euler_angles

def calculate_hip_angles(vectors, coordinate_axes):
    """
    Calculate the signed angles between the projection of vectors and the three coordinate planes (XY, YZ, XZ).

    Args:
        vectors (np.ndarray): Shape (n_frame, 3), the vectors whose angles are computed.
        coordinate_axes (np.ndarray): Shape (n_frame, 3, 3), custom coordinate system per frame.

    Returns:
        np.ndarray: Signed angles (n_frame, 3), in degrees, where columns represent angles to XY, YZ, and XZ planes.
    """
    n_frame = vectors.shape[0]

    # Normalize the coordinate axes to ensure they are proper rotation matrices
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)

    # Transform vectors into the new coordinate system
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    # Compute signed angles with respect to coordinate planes
    angles_to_planes = np.zeros((n_frame, 3))
    angles_to_neg_y = np.zeros((n_frame, 3))
    for i in range(n_frame):
        v = vectors_transformed[i] / np.linalg.norm(vectors_transformed[i])
        v_xy = np.array([v[0], v[1]])  # (x, y)
        if np.linalg.norm(v_xy) > 1e-8:  # Avoid division by zero
            v_xy /= np.linalg.norm(v_xy)
            angles_to_neg_y[i, 0] = np.degrees(np.arctan2(v_xy[0], -v_xy[1]))  # Angle with negative Y-axis

        # Compute the projection on YZ plane
        v_yz = np.array([v[1], v[2]])  # (y, z)
        if np.linalg.norm(v_yz) > 1e-8:  # Avoid division by zero
            v_yz /= np.linalg.norm(v_yz)
            angles_to_neg_y[i, 1] = np.degrees(np.arctan2(v_yz[1], -v_yz[0]))

    return angles_to_neg_y

def calculate_lumbar_angles(vectors, coordinate_axes):
    """
    Calculate the signed angles between vectors and the three coordinate planes (XY, YZ, XZ).

    Args:
        vectors (np.ndarray): Shape (n_frame, 3), the vectors whose angles are computed.
        coordinate_axes (np.ndarray): Shape (n_frame, 3, 3), custom coordinate system per frame.

    Returns:
        np.ndarray: Signed angles (n_frame, 3), in degrees, where columns represent angles to XY, YZ, and XZ planes.
    """
    n_frame = vectors.shape[0]

    # Normalize the coordinate axes to ensure they are proper rotation matrices
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)

    # Transform vectors into the new coordinate system
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    # Compute signed angles with respect to coordinate planes
    angles_to_planes = np.zeros((n_frame, 3))
    angles_to_neg_y = np.zeros((n_frame, 3))
    for i in range(n_frame):
        v = vectors_transformed[i] / np.linalg.norm(vectors_transformed[i])
        v_xy = np.array([v[0], v[1]])  # (x, y)
        if np.linalg.norm(v_xy) > 1e-8:  # Avoid division by zero
            v_xy /= np.linalg.norm(v_xy)
            angles_to_neg_y[i, 0] = np.degrees(np.arctan2(v_xy[0], v_xy[1]))  # Angle with positive Y-axis

        # Compute the projection on YZ plane
        v_yz = np.array([v[1], v[2]])  # (y, z)
        if np.linalg.norm(v_yz) > 1e-8:  # Avoid division by zero
            v_yz /= np.linalg.norm(v_yz)
            angles_to_neg_y[i, 1] = np.degrees(np.arctan2(v_yz[1], v_yz[0]))

    return angles_to_neg_y

def calculate_neck_angles(vectors, coordinate_axes):
    """
    Calculate the signed angles between vectors and the three coordinate planes (XY, YZ, XZ).

    Args:
        vectors (np.ndarray): Shape (n_frame, 3), the vectors whose angles are computed.
        coordinate_axes (np.ndarray): Shape (n_frame, 3, 3), custom coordinate system per frame.

    Returns:
        np.ndarray: Signed angles (n_frame, 3), in degrees, where columns represent angles to XY, YZ, and XZ planes.
    """
    n_frame = vectors.shape[0]

    # Normalize the coordinate axes to ensure they are proper rotation matrices
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)

    # Transform vectors into the new coordinate system
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    # Compute signed angles with respect to coordinate planes
    angles_to_planes = np.zeros((n_frame, 3))
    angles_to_neg_y = np.zeros((n_frame, 3))
    for i in range(n_frame):
        v = vectors_transformed[i] / np.linalg.norm(vectors_transformed[i])
        v_xy = np.array([v[0], v[1]])  # (x, y)
        if np.linalg.norm(v_xy) > 1e-8:  # Avoid division by zero
            v_xy /= np.linalg.norm(v_xy)
            angles_to_neg_y[i, 0] = np.degrees(np.arctan2(v_xy[0], v_xy[1]))  # Angle with positive Y-axis

        # Compute the projection on YZ plane to y-axis angle
        v_yz = np.array([v[1], v[2]])  # (y, z)
        if np.linalg.norm(v_yz) > 1e-8:  # Avoid division by zero
            v_yz /= np.linalg.norm(v_yz)
            angles_to_neg_y[i, 1] = np.degrees(np.arctan2(v_yz[1], v_yz[0]))

    return angles_to_neg_y

def calculate_shoulder_angles(vectors, coordinate_axes):
    """
    Calculate the signed angles between vectors and the three coordinate planes (XY, YZ, XZ).

    Args:
        vectors (np.ndarray): Shape (n_frame, 3), the vectors whose angles are computed.
        coordinate_axes (np.ndarray): Shape (n_frame, 3, 3), custom coordinate system per frame.

    Returns:
        np.ndarray: Signed angles (n_frame, 3), in degrees, where columns represent angles to XY, YZ, and XZ planes.
    """
    n_frame = vectors.shape[0]

    # Normalize the coordinate axes to ensure they are proper rotation matrices
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)

    # Transform vectors into the new coordinate system
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    # Compute signed angles with respect to coordinate planes
    angles_to_planes = np.zeros((n_frame, 3))
    angles_to_neg_y = np.zeros((n_frame, 3))
    for i in range(n_frame):
        v = vectors_transformed[i] / np.linalg.norm(vectors_transformed[i])
        v_xy = np.array([v[0], v[1]])  # (x, y)
        if np.linalg.norm(v_xy) > 1e-8:  # Avoid division by zero
            v_xy /= np.linalg.norm(v_xy)
            angles_to_neg_y[i, 0] = np.degrees(np.arctan2(v_xy[0], -v_xy[1]))  # Angle with negative Y-axis

        # Compute the projection on YZ plane
        v_yz = np.array([v[1], v[2]])  # (y, z)
        if np.linalg.norm(v_yz) > 1e-8:  # Avoid division by zero
            v_yz /= np.linalg.norm(v_yz)
            angles_to_neg_y[i, 1] = np.degrees(np.arctan2(v_yz[1], -v_yz[0]))

    return angles_to_neg_y

def compute_signed_angle_xy_projection(vectors, coordinate_axes):
    """
    Compute the signed angle between the vector and its projection on the XY plane.
    - Positive if the vector's z-component is positive.
    - Negative if the vector's z-component is negative.

    Args:
        vectors (np.ndarray): Shape (n_frame, 3), the input vectors.
        coordinate_axes (np.ndarray): Shape (n_frame, 3, 3), transformation matrices.

    Returns:
        np.ndarray: Signed angles in degrees, shape (n_frame,).
    """
    n_frame = vectors.shape[0]

    # Normalize coordinate axes (ensure they are unit vectors)
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)

    # Transform vectors into the new coordinate system
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    # Compute the projection onto the XY plane (setting Z-component to zero)
    projections_xy = np.copy(vectors_transformed)
    projections_xy[:, 2] = 0  # Set the Z component to zero

    # Compute norms
    vector_norms = np.linalg.norm(vectors_transformed, axis=1)
    projection_norms = np.linalg.norm(projections_xy, axis=1)

    # Avoid division by zero by setting norms close to zero to NaN
    valid_mask = projection_norms > 1e-8
    angles = np.full(n_frame, np.nan)  # Initialize with NaN

    # Compute angles where the projection norm is not zero
    if np.any(valid_mask):
        cos_theta = np.clip(
            np.sum(vectors_transformed[valid_mask] * projections_xy[valid_mask], axis=1) /
            (vector_norms[valid_mask] * projection_norms[valid_mask]),
            -1.0, 1.0
        )
        angles[valid_mask] = np.degrees(np.arccos(cos_theta))

        # Adjust sign based on the z-component of the transformed vector
        angles[valid_mask] *= np.sign(vectors_transformed[valid_mask, 2])

    return angles

def compute_hip_rotation(hip_angles, hip_positions, knee_positions, ankle_positions, body_coordinate):
    """
    Compute the hip rotation angle using hip flexion and adduction to derive the rotated coordinate frame.

    Args:
        hip_angles (np.ndarray): shape (n_frames, 2), hip_flexion and hip_adduction in degrees.
        hip_positions (np.ndarray): shape (n_frames, 3)
        knee_positions (np.ndarray): shape (n_frames, 3)
        ankle_positions (np.ndarray): shape (n_frames, 3)
        body_coordinate (np.ndarray): shape (n_frames, 3, 3)

    Returns:
        np.ndarray: shape (n_frames,), hip rotation angle in degrees
    """
    n_frames = hip_angles.shape[0]
    hip_rotation_angles = np.zeros(n_frames)
    vectors = ankle_positions - knee_positions
    vectors_transformed = np.einsum('nij,nj->ni', body_coordinate, vectors)

    for i in range(n_frames):
        flexion = np.radians(hip_angles[i, 0])
        adduction = np.radians(-hip_angles[i, 1])

        # Original coordinate system at hip (from body)
        body_axes = body_coordinate[i].T  # shape (3, 3)
        axis_z = body_axes[:, 2]  # z-axis of body

        # Step 2: create rotation objects for each step
        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()  # rotate around body's z-axis
        R_knee_flexion = rot1 @ body_axes
        # knee_flexion_axes = R_knee_flexion.T

        axis_x = R_knee_flexion[:, 0]  # x-axis of body
        rot2 = R.from_rotvec(adduction * axis_x).as_matrix()  # then around body's x-axis, here maybe the problem, should rotate based on axis_x after flexion
        R_knee = rot2 @ R_knee_flexion
        knee_axes = R_knee.T
        # Compose knee coordinate
        # knee_axes = R_adduction @ R_flexion @ body_axes  # shape (3, 3)
        # knee_axes = body_axes @ R_flexion @ R_adduction

        # Compute vector from ankle to knee in body frame
        ankle_knee_vec = ankle_positions[i] - knee_positions[i]
        # ankle_knee_in_body = body_axes @ ankle_knee_vec  # (3,)

        # Transform this vector into the knee coordinate
        ankle_knee_in_knee_coord = knee_axes @ ankle_knee_vec

        # Project it onto XY plane
        proj_xy = np.copy(ankle_knee_in_knee_coord)
        proj_xy[2] = 0

        norm_vec = np.linalg.norm(ankle_knee_in_knee_coord)
        norm_proj = np.linalg.norm(proj_xy)

        if norm_proj < 1e-8 or norm_vec < 1e-8:
            hip_rotation_angles[i] = 0
        else:
            dot = np.dot(ankle_knee_in_knee_coord, proj_xy)
            angle = np.degrees(np.arccos(np.clip(dot / (norm_vec * norm_proj), -1.0, 1.0)))
            sign = np.sign(ankle_knee_in_knee_coord[2])  # Positive if pointing upward
            hip_rotation_angles[i] = angle * sign

    return hip_rotation_angles

def compute_lumbar_rotation(lumbar_angles, neck_positions, shoulder_positions, body_coordinate):
    """
    Compute lumbar rotation from neck to shoulder vector under rotated coordinate system.

    Args:
        lumbar_angles (np.ndarray): shape (n_frames, 2), lumbar flexion and adduction(bending) in degrees.
        neck_positions (np.ndarray): shape (n_frames, 3)
        shoulder_positions (np.ndarray): shape (n_frames, 3)
        body_coordinate (np.ndarray): shape (n_frames, 3, 3)

    Returns:
        np.ndarray: shape (n_frames,), lumbar rotation in degrees
    """
    n_frames = lumbar_angles.shape[0]
    rotation_angles = np.zeros(n_frames)
    neck_coordinate = []

    for i in range(n_frames):
        flexion = np.radians(-lumbar_angles[i, 0])   # rotation around z
        adduction = np.radians(lumbar_angles[i, 1]) # rotation around x

        body_axes = body_coordinate[i].T  # shape (3, 3)
        axis_z = body_axes[:, 2]  # z-axis of body
        # Step 2: create rotation objects for each step
        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()  # rotate around body's z-axis
        R_lumbar_flexion = rot1 @ body_axes
        axis_x = R_lumbar_flexion[:, 0]  # x-axis of body
        rot2 = R.from_rotvec(adduction * axis_x).as_matrix()
        R_lumbar = rot2 @ R_lumbar_flexion
        neck_axes = R_lumbar.T
        neck_coordinate.append(neck_axes)

        # Vector from neck to right shoulder
        neck_shoulder_vec = shoulder_positions[i] - neck_positions[i]
        vec_in_neck = neck_axes @ neck_shoulder_vec

        # Project onto YZ plane
        proj_yz = np.copy(vec_in_neck)
        proj_yz[0] = 0

        norm_vec = np.linalg.norm(vec_in_neck)
        norm_proj = np.linalg.norm(proj_yz)

        if norm_vec < 1e-8 or norm_proj < 1e-8:
            rotation_angles[i] = 0
        else:
            dot = np.dot(vec_in_neck, proj_yz)
            angle = np.degrees(np.arccos(np.clip(dot / (norm_vec * norm_proj), -1.0, 1.0)))
            sign = np.sign(vec_in_neck[0])  # signed wrt X-axis
            rotation_angles[i] = angle * sign
            # rotation_angles[i] = angle
    neck_coordinate = np.array(neck_coordinate)
    return rotation_angles, neck_coordinate

def compute_shoulder_rotation(shoulder_angles, elbow_positions, wrist_positions, shoulder_coordinate):
    """
    Compute lumbar rotation from neck to shoulder vector under rotated coordinate system.

    Args:
        lumbar_angles (np.ndarray): shape (n_frames, 2), lumbar flexion and adduction(bending) in degrees.
        neck_positions (np.ndarray): shape (n_frames, 3)
        shoulder_positions (np.ndarray): shape (n_frames, 3)
        body_coordinate (np.ndarray): shape (n_frames, 3, 3)

    Returns:
        np.ndarray: shape (n_frames,), lumbar rotation in degrees
    """
    n_frames = shoulder_angles.shape[0]
    rotation_angles = np.zeros(n_frames)
    # neck_coordinate = []

    for i in range(n_frames):
        flexion = np.radians(shoulder_angles[i, 0])   # rotation around z
        adduction = np.radians(-shoulder_angles[i, 1]) # rotation around x

        shoulder_axes = shoulder_coordinate[i].T  # shape (3, 3)
        axis_z = shoulder_axes[:, 2]  # z-axis of body

        # Step 2: create rotation objects for each step
        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()  # rotate around body's z-axis
        R_shoulder_flexion = rot1 @ shoulder_axes
        axis_x = R_shoulder_flexion[:, 0]  # x-axis of body
        rot2 = R.from_rotvec(adduction * axis_x).as_matrix()
        R_shoulder = rot2 @ R_shoulder_flexion
        shoulder_axes = R_shoulder.T
        # neck_coordinate.append(neck_axes)

        # Vector from neck to right shoulder
        elbow_wrist_vec = wrist_positions[i] - elbow_positions[i]
        vec_in_shoulder = shoulder_axes @ elbow_wrist_vec

        # Project onto XY plane
        proj_xy = np.copy(vec_in_shoulder)
        proj_xy[2] = 0

        norm_vec = np.linalg.norm(vec_in_shoulder)
        norm_proj = np.linalg.norm(proj_xy)

        if norm_vec < 1e-8 or norm_proj < 1e-8:
            rotation_angles[i] = 0
        else:
            dot = np.dot(vec_in_shoulder, proj_xy)
            angle = np.degrees(np.arccos(np.clip(dot / (norm_vec * norm_proj), -1.0, 1.0)))
            sign = np.sign(vec_in_shoulder[2])  # signed wrt X-axis
            rotation_angles[i] = angle * sign
            # rotation_angles[i] = angle
    #neck_coordinate = np.array(neck_coordinate)
    return rotation_angles

def compute_shoulder_coordinate(body_coordinate, lumbar_flexion, lumbar_adduction, lumbar_rotation, neck_coordinate):
    n_frames = body_coordinate.shape[0]
    shoulder_coordinate = []

    for i in range(n_frames):
        flexion = np.radians(-lumbar_flexion[i])  # rotation around x
        adduction = np.radians(lumbar_adduction[i])  # rotation around z
        rotation = np.radians(lumbar_rotation[i])
        # Step 2: create rotation objects for each step
        body_axes = body_coordinate[i].T  # shape (3, 3)
        axis_z = body_axes[:, 2]  # z-axis of body
        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()  # rotate around body's z-axis
        R_body_flexion = rot1 @ body_axes
        axis_x = R_body_flexion[:, 0]  # x-axis of body
        rot2 = R.from_rotvec(adduction * axis_x).as_matrix()
        R_body_flexion_addcution = rot2 @ R_body_flexion
        axis_y = R_body_flexion_addcution[:, 1]
        rot3 = R.from_rotvec(rotation * axis_y).as_matrix()  # then around body's x-axis
        R_shoulder = rot3 @ R_body_flexion_addcution
        shoulder_axis = R_shoulder.T
        shoulder_coordinate.append(shoulder_axis)
    shoulder_coordinate = np.array(shoulder_coordinate)
    return shoulder_coordinate

# without hand position, so unusable
def compute_elbow_coordinate(shoulder_coordinate, shoulder_flexion, shoulder_adduction, shoulder_rotation):
    n_frames = shoulder_coordinate.shape[0]
    elbow_coordinate = []

    for i in range(n_frames):
        flexion = np.radians(shoulder_flexion[i])  # rotation around x
        adduction = np.radians(-shoulder_adduction[i])  # rotation around z
        rotation = np.radians(-shoulder_rotation[i])

        shoulder_axes = shoulder_coordinate[i].T  # shape (3, 3)
        axis_z = shoulder_axes[:, 2]  # z-axis of body
        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()  # rotate around body's z-axis
        R_shoulder_flexion = rot1 @ shoulder_axes
        axis_x = R_shoulder_flexion[:, 0]  # x-axis of body
        rot2 = R.from_rotvec(adduction * axis_x).as_matrix()
        R_shoulder_flexion_addcution = rot2 @ R_shoulder_flexion
        axis_y = R_shoulder_flexion_addcution[:, 1]
        rot3 = R.from_rotvec(rotation * axis_y).as_matrix()  # then around body's x-axis
        R_elbow = rot3 @ R_shoulder_flexion_addcution
        elbow_axis = R_elbow.T
        elbow_coordinate.append(elbow_axis)
    elbow_coordinate = np.array(elbow_coordinate)
    return elbow_coordinate

def compute_elbow_rotation(elbow_angle, hand_index_mcp, hand_middle_mcp, elbow_coordinate, wrist_elbow_normed):
    """
    Compute lumbar rotation from neck to shoulder vector under rotated coordinate system.

    Args:
        lumbar_angles (np.ndarray): shape (n_frames, 2), lumbar flexion and adduction(bending) in degrees.
        neck_positions (np.ndarray): shape (n_frames, 3)
        shoulder_positions (np.ndarray): shape (n_frames, 3)
        body_coordinate (np.ndarray): shape (n_frames, 3, 3)

    Returns:
        np.ndarray: shape (n_frames,), lumbar rotation in degrees
    """
    n_frames = elbow_angle.shape[0]
    rotation_angles = np.zeros(n_frames)
    elbow_flexion_coordinate = []

    for i in range(n_frames):
        flexion = np.radians(elbow_angle[i])   # rotation around z
        # adduction = np.radians(-shoulder_angles[i, 1]) # rotation around x

        elbow_axes = elbow_coordinate[i].T  # shape (3, 3)
        axis_z = elbow_axes[:, 2]  # z-axis of body
        # Step 2: create rotation objects for each step
        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()  # rotate around body's z-axis
        R_elbow = rot1 @ elbow_axes
        elbow_axes = R_elbow.T
        elbow_flexion_coordinate.append(elbow_axes)

        # Vector from neck to right shoulder
        middle_index_vec = hand_index_mcp[i] - hand_middle_mcp[i]
        vec_in_elbow = elbow_axes @ middle_index_vec

        # Project onto YZ plane
        proj_yz = np.copy(vec_in_elbow)
        proj_yz[0] = 0

        norm_vec = np.linalg.norm(vec_in_elbow)
        norm_proj = np.linalg.norm(proj_yz)

        if norm_vec < 1e-8 or norm_proj < 1e-8:
            rotation_angles[i] = 0
        else:
            dot = np.dot(vec_in_elbow, proj_yz)
            angle = np.degrees(np.arccos(np.clip(dot / (norm_vec * norm_proj), -1.0, 1.0)))
            sign = np.sign(vec_in_elbow[0])  # signed wrt X-axis
            rotation_angles[i] = angle * sign
            # rotation_angles[i] = angle
    elbow_flexion_coordinate = np.array(elbow_flexion_coordinate)
    return rotation_angles, elbow_flexion_coordinate

def compute_wrist_coordinate(elbow_coordinate, elbow_rotation):
    n_frames = elbow_coordinate.shape[0]
    wrist_coordinate = []
    for i in range(n_frames):
        rotation = np.radians(elbow_rotation[i])  # rotation around y

        elbow_axes = elbow_coordinate[i].T  # shape (3, 3)
        axis_y = elbow_axes[:, 1]  # y-axis of body
        rot1 = R.from_rotvec(rotation * axis_y).as_matrix()  # rotate around body's z-axis
        R_elbow_rotation = rot1 @ elbow_axes
        # axis_x = R_shoulder_flexion[:, 0]  # x-axis of body
        # rot2 = R.from_rotvec(adduction * axis_x).as_matrix()
        # R_shoulder_flexion_addcution = rot2 @ R_shoulder_flexion
        # axis_y = R_shoulder_flexion_addcution[:, 1]
        # rot3 = R.from_rotvec(rotation * axis_y).as_matrix()  # then around body's x-axis
        # R_elbow = rot3 @ R_shoulder_flexion_addcution
        wrist_axis = R_elbow_rotation.T
        wrist_coordinate.append(wrist_axis)
    wrist_coordinate = np.array(wrist_coordinate)
    return wrist_coordinate

def calculate_wrist_angles(vectors, coordinate_axes):
    """
    Calculate the signed angles between vectors and the three coordinate planes (XY, YZ, XZ).

    Args:
        vectors (np.ndarray): Shape (n_frame, 3), the vectors whose angles are computed.
        coordinate_axes (np.ndarray): Shape (n_frame, 3, 3), custom coordinate system per frame.

    Returns:
        np.ndarray: Signed angles (n_frame, 3), in degrees, where columns represent angles to XY, YZ, and XZ planes.
    """
    n_frame = vectors.shape[0]

    # Normalize the coordinate axes to ensure they are proper rotation matrices
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)

    # Transform vectors into the new coordinate system
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    # Compute signed angles with respect to coordinate planes
    angles_to_planes = np.zeros((n_frame, 3))
    angles_to_neg_y = np.zeros((n_frame, 3))
    for i in range(n_frame):
        v = vectors_transformed[i] / np.linalg.norm(vectors_transformed[i])
        v_xy = np.array([v[0], v[1]])  # (x, y)
        if np.linalg.norm(v_xy) > 1e-8:  # Avoid division by zero
            v_xy /= np.linalg.norm(v_xy)
            angles_to_neg_y[i, 0] = np.degrees(np.arctan2(v_xy[0], -v_xy[1]))  # Angle with negative Y-axis

        # Compute the projection on YZ plane
        v_yz = np.array([v[1], v[2]])  # (y, z)
        if np.linalg.norm(v_yz) > 1e-8:  # Avoid division by zero
            v_yz /= np.linalg.norm(v_yz)
            angles_to_neg_y[i, 1] = np.degrees(np.arctan2(v_yz[1], -v_yz[0]))

    return angles_to_neg_y

def calculate_knee_angle(hip, knee, ankle):
    thigh_vector = knee - hip  # shape: (n_frames, 3)
    shank_vector = ankle - knee  # shape: (n_frames, 3)
    # Dot product for each frame
    dot_product = np.einsum('ij,ij->i', thigh_vector, shank_vector)  # shape: (n_frames,)
    # Magnitudes of vectors
    magnitude_thigh = np.linalg.norm(thigh_vector, axis=1)  # shape: (n_frames,)
    magnitude_shank = np.linalg.norm(shank_vector, axis=1)  # shape: (n_frames,)
    # Prevent division by zero
    denominator = np.clip(magnitude_thigh * magnitude_shank, 1e-8, None)
    # Calculate the angle in radians
    angle_rad = np.arccos(np.clip(dot_product / denominator, -1.0, 1.0))  # shape: (n_frames,)
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def calculate_ankle_angle(knee, ankle, foot):
    """
    Calculate the ankle joint angle based on the coordinates of the knee, ankle, and foot.

    Parameters:
    knee (numpy.ndarray): (n_frames, 3) array of knee coordinates
    ankle (numpy.ndarray): (n_frames, 3) array of ankle coordinates
    foot (numpy.ndarray): (n_frames, 3) array of foot coordinates

    Returns:
    numpy.ndarray: (n_frames,) array of ankle joint angles in degrees
    """
    shank_vector = ankle - knee  # shape: (n_frames, 3)
    foot_vector = foot - ankle  # shape: (n_frames, 3)

    # Dot product for each frame
    dot_product = np.einsum('ij,ij->i', shank_vector, foot_vector)  # shape: (n_frames,)

    # Magnitudes of vectors
    magnitude_shank = np.linalg.norm(shank_vector, axis=1)  # shape: (n_frames,)
    magnitude_foot = np.linalg.norm(foot_vector, axis=1)  # shape: (n_frames,)

    # Prevent division by zero
    denominator = np.clip(magnitude_shank * magnitude_foot, 1e-8, None)

    # Calculate the angle in radians
    angle_rad = np.arccos(np.clip(dot_product / denominator, -1.0, 1.0))  # shape: (n_frames,)

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def calculate_elbow_angle(shoulder, elbow, wrist):
    thigh_vector = elbow - shoulder  # shape: (n_frames, 3)
    shank_vector = wrist - elbow  # shape: (n_frames, 3)
    # Dot product for each frame
    dot_product = np.einsum('ij,ij->i', thigh_vector, shank_vector)  # shape: (n_frames,)
    # Magnitudes of vectors
    magnitude_thigh = np.linalg.norm(thigh_vector, axis=1)  # shape: (n_frames,)
    magnitude_shank = np.linalg.norm(shank_vector, axis=1)  # shape: (n_frames,)
    # Prevent division by zero
    denominator = np.clip(magnitude_thigh * magnitude_shank, 1e-8, None)
    # Calculate the angle in radians
    angle_rad = np.arccos(np.clip(dot_product / denominator, -1.0, 1.0))  # shape: (n_frames,)
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def compute_hand_coordinate(wrist, joint1, joint2, left_hand=False, pinky=False, smooth_forward=False):
    # across = right_hip - left_hip  # Hip width direction
    # across2 = right_shoulder - left_shoulder
    # across = across1 + across2

    forward = joint1 - wrist
    v2 = joint2 - joint1
    if pinky:
        v2 = -v2

    forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)

    # Remove the component of v2 along x -> perpendicular part
    dot = np.sum(v2 * forward, axis=-1, keepdims=True)
    v2_perp = v2 - dot * forward
    # v2_perp = v2 - np.dot(v2, forward) * forward
    across = v2_perp/ np.linalg.norm(v2_perp, axis=-1, keepdims=True)  # this is the requested z-axis
    if not left_hand:
        across = -across

    # y to make it right-handed
    up = np.cross(across, forward)
    up = up/ np.linalg.norm(up, axis=-1, keepdims=True)

    if smooth_forward:
        forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')

    human_coordinate = np.array([forward, up, across])
    combined_coordinates = np.transpose(human_coordinate, (1, 0, 2))
    return combined_coordinates

def calculate_flexion_angles(vectors, coordinate_axes):
        """
        Calculate the signed angles between vectors and the three coordinate planes (XY, YZ, XZ).

        Args:
            vectors (np.ndarray): Shape (n_frame, 3), the vectors whose angles are computed.
            coordinate_axes (np.ndarray): Shape (n_frame, 3, 3), custom coordinate system per frame.

        Returns:
            np.ndarray: Signed angles (n_frame, 3), in degrees, where columns represent angles to XY, YZ, and XZ planes.
        """
        n_frame = vectors.shape[0]

        # Normalize the coordinate axes to ensure they are proper rotation matrices
        coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)

        # Transform vectors into the new coordinate system
        vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

        # Compute signed angles with respect to coordinate planes
        # angles_to_planes = np.zeros((n_frame, 3))
        angles_to_pos_x = np.zeros((n_frame, 3))
        for i in range(n_frame):
            v = vectors_transformed[i] / np.linalg.norm(vectors_transformed[i])
            v_xy = np.array([v[0], v[1]])  # (x, y)
            if np.linalg.norm(v_xy) > 1e-8:  # Avoid division by zero
                v_xy /= np.linalg.norm(v_xy)
                angles_to_pos_x[i, 0] = np.degrees(np.arctan2(v_xy[1], v_xy[0]))  # Angle with positive X-axis

            # # Compute the projection on YZ plane
            # v_yz = np.array([v[1], v[2]])  # (y, z)
            # if np.linalg.norm(v_yz) > 1e-8:  # Avoid division by zero
            #     v_yz /= np.linalg.norm(v_yz)
            #     angles_to_neg_y[i, 1] = np.degrees(np.arctan2(v_yz[1], v_yz[0]))

        return angles_to_pos_x

def compute_finger_coordinate(parent_coordinate, rot_flexion, rot_adduction):
    # for finger's coordinate rotation, it is different with body, should rotate along z-axis then new y-axis
    n_frames = parent_coordinate.shape[0]
    child_coordinate = []

    for i in range(n_frames):
        flexion = np.radians(rot_flexion[i])  # rotation around z
        adduction = np.radians(-rot_adduction[i])  # rotation around x
        # rotation = np.radians(lumbar_rotation[i])
        # Step 2: create rotation objects for each step
        body_axes = parent_coordinate[i].T  # shape (3, 3)
        axis_z = body_axes[:, 2]  # z-axis of body
        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()  # rotate around body's z-axis
        R_body_flexion = rot1 @ body_axes
        axis_y = R_body_flexion[:, 1]  # y-axis of body
        rot2 = R.from_rotvec(adduction * axis_y).as_matrix()
        R_body_flexion_addcution = rot2 @ R_body_flexion
        # axis_y = R_body_flexion_addcution[:, 1]
        # rot3 = R.from_rotvec(rotation * axis_y).as_matrix()  # then around body's x-axis
        # R_shoulder = rot3 @ R_body_flexion_addcution
        child_axis = R_body_flexion_addcution.T
        child_coordinate.append(child_axis)
    child_coordinate = np.array(child_coordinate)
    return child_coordinate

def calculate_finger_angle(pip_mcp, dip_pip):
    # thigh_vector = pip - shoulder  # shape: (n_frames, 3)
    # shank_vector = wrist - elbow  # shape: (n_frames, 3)
    # Dot product for each frame
    dot_product = np.einsum('ij,ij->i', pip_mcp, dip_pip)  # shape: (n_frames,)
    # Magnitudes of vectors
    magnitude_pip_mcp = np.linalg.norm(pip_mcp, axis=1)  # shape: (n_frames,)
    magnitude_dip_pip = np.linalg.norm(dip_pip, axis=1)  # shape: (n_frames,)
    # Prevent division by zero
    denominator = np.clip(magnitude_pip_mcp * magnitude_dip_pip, 1e-8, None)
    # Calculate the angle in radians
    angle_rad = np.arccos(np.clip(dot_product / denominator, -1.0, 1.0))  # shape: (n_frames,)
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg
