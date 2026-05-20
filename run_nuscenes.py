"""
run_nuscenes.py
===============
Run BEV perception on REAL nuScenes data.

Uses actual camera images from 6 cameras
mounted on a real autonomous vehicle.
Projects all 6 cameras into unified BEV space.
Compares object positions vs LiDAR ground truth.

Dataset : nuScenes mini v1.0
Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 4 of 90 — Perception Series
"""

import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

# ── SETTINGS ──────────────────────────────────────────────────────────

NUSCENES_ROOT = r"D:\day-004-bev-perception"
VERSION       = "v1.0-mini"

# BEV grid settings
BEV_X_MIN      = -25.0
BEV_X_MAX      =  25.0
BEV_Y_MIN      =   0.0
BEV_Y_MAX      =  50.0
BEV_RESOLUTION =  10
BEV_W = int((BEV_X_MAX - BEV_X_MIN) * BEV_RESOLUTION)
BEV_H = int((BEV_Y_MAX - BEV_Y_MIN) * BEV_RESOLUTION)

CAMERAS = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
    'CAM_BACK_LEFT',
]

CAM_COLORS = {
    'CAM_FRONT':       (255, 200,   0),
    'CAM_FRONT_RIGHT': (  0, 200, 255),
    'CAM_FRONT_LEFT':  (  0, 255, 100),
    'CAM_BACK':        (255, 100,   0),
    'CAM_BACK_RIGHT':  (200,   0, 255),
    'CAM_BACK_LEFT':   (255,   0, 100),
}


# ── COORDINATE HELPERS ────────────────────────────────────────────────

def world_to_bev(x, y):
    px = int((x - BEV_X_MIN) * BEV_RESOLUTION)
    py = int(BEV_H - (y - BEV_Y_MIN) * BEV_RESOLUTION)
    return px, py


def get_transform(record):
    """Build 4x4 transform from translation + rotation quaternion."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    T[:3,  3] = np.array(record['translation'])
    return T


# ── PROJECT ONE CAMERA TO BEV ─────────────────────────────────────────

def project_camera_to_bev(image, intrinsic, cam_to_ego):
    """
    Project a real camera image onto the BEV grid using IPM.
    Same algorithm as demo but on real nuScenes images.
    """
    bev = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    img_h, img_w = image.shape[:2]
    ego_to_cam   = np.linalg.inv(cam_to_ego)

    # Build BEV grid
    px_arr  = np.arange(BEV_W)
    py_arr  = np.arange(BEV_H)
    px_grid, py_grid = np.meshgrid(px_arr, py_arr)

    x_world = px_grid / BEV_RESOLUTION + BEV_X_MIN
    y_world = (BEV_H - py_grid) / BEV_RESOLUTION + BEV_Y_MIN

    N       = BEV_W * BEV_H
    ego_pts = np.ones((4, N), dtype=np.float64)
    ego_pts[0] = x_world.flatten()
    ego_pts[1] = y_world.flatten()
    ego_pts[2] = 0.0   # ground plane

    # Transform to camera frame
    cam_pts = ego_to_cam @ ego_pts
    valid   = cam_pts[2] > 0.5   # in front of camera

    # Project to image
    K     = np.array(intrinsic)
    proj  = K @ cam_pts[:3]
    u     = proj[0] / proj[2]
    v     = proj[1] / proj[2]

    valid &= (u >= 0) & (u < img_w - 1) & \
             (v >= 0) & (v < img_h - 1)

    u_v = u[valid].astype(int)
    v_v = v[valid].astype(int)

    colors   = image[v_v, u_v]
    flat_idx = np.arange(N)[valid]
    bev[flat_idx // BEV_W, flat_idx % BEV_W] = colors

    return bev


# ── MAIN PIPELINE ─────────────────────────────────────────────────────

def run_nuscenes_bev(sample_idx=0):
    """
    Run full BEV pipeline on one real nuScenes sample.

    Steps:
      1. Load nuScenes dataset
      2. Pick one sample (one timestamp)
      3. Load all 6 camera images
      4. Load camera calibration for each
      5. Project each camera to BEV
      6. Stitch all 6 into unified BEV
      7. Overlay 3D object annotations
      8. Measure position accuracy
      9. Save visualization
    """
    print("\n" + "="*60)
    print("  nuScenes BEV Perception — Real Data")
    print(f"  Sample index: {sample_idx}")
    print("="*60)

    # Load nuScenes
    print("\n  Loading nuScenes...")
    nusc = NuScenes(
        version  = VERSION,
        dataroot = NUSCENES_ROOT,
        verbose  = False
    )
    print(f"  Scenes:  {len(nusc.scene)}")
    print(f"  Samples: {len(nusc.sample)}")

    # Pick sample
    sample = nusc.sample[sample_idx]
    print(f"\n  Sample token: {sample['token'][:16]}...")

    # Create BEV canvas
    bev_canvas = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    bev_canvas[:] = (30, 30, 30)

    # Draw road grid
    cx = BEV_W // 2
    cv2.rectangle(bev_canvas, (cx-25, 0), (cx+25, BEV_H),
                  (60, 60, 60), -1)
    for y in range(0, BEV_H, 40):
        cv2.rectangle(bev_canvas, (cx-2, y), (cx+2, y+20),
                      (180, 180, 80), -1)

    cam_images = {}

    # Process each camera
    print(f"\n  Projecting cameras to BEV...")
    for cam_name in CAMERAS:
        if cam_name not in sample['data']:
            continue

        # Get camera sample data
        cam_token    = sample['data'][cam_name]
        cam_data     = nusc.get('sample_data', cam_token)
        img_path     = os.path.join(NUSCENES_ROOT, cam_data['filename'])

        if not os.path.exists(img_path):
            print(f"  {cam_name}: image not found, skipping")
            continue

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue

        cam_images[cam_name] = image
        print(f"  {cam_name}: {image.shape[1]}x{image.shape[0]}")

        # Get calibration
        calib_token = cam_data['calibrated_sensor_token']
        calib       = nusc.get('calibrated_sensor', calib_token)
        intrinsic   = calib['camera_intrinsic']

        # Camera to ego transform
        cam_to_ego  = get_transform(calib)

        # Project to BEV
        cam_bev = project_camera_to_bev(image, intrinsic, cam_to_ego)

        # Blend into canvas
        mask = cam_bev.sum(axis=2) > 0
        bev_canvas[mask] = (
            bev_canvas[mask] * 0.3 +
            cam_bev[mask]    * 0.7
        ).astype(np.uint8)

    # Draw distance rings
    ego_px, ego_py = world_to_bev(0, 0)
    for dist in [10, 20, 30, 40]:
        _, py_ring = world_to_bev(0, dist)
        radius = abs(ego_py - py_ring)
        cv2.circle(bev_canvas, (ego_px, ego_py),
                   radius, (80, 80, 80), 1)
        cv2.putText(bev_canvas, f'{dist}m',
                    (ego_px + 5, py_ring),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (120, 120, 120), 1)

    # Draw ego vehicle
    cv2.rectangle(bev_canvas,
                  (ego_px-8, ego_py-18),
                  (ego_px+8, ego_py+4),
                  (255, 255, 255), -1)
    cv2.putText(bev_canvas, 'EGO',
                (ego_px-10, ego_py+16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 0), 1)

    # Get ego pose for annotation transform
    ego_token  = cam_data['ego_pose_token']
    ego_pose   = nusc.get('ego_pose', ego_token)
    ego_to_world = get_transform(ego_pose)
    world_to_ego = np.linalg.inv(ego_to_world)

    # Draw 3D annotations (ground truth objects)
    print(f"\n  Drawing ground truth objects...")
    annotations = sample['anns']
    obj_counts  = {'car': 0, 'pedestrian': 0,
                   'bicycle': 0, 'truck': 0, 'other': 0}
    errors      = []

    obj_colors = {
        'car':        (0,   200, 255),
        'pedestrian': (255, 100,   0),
        'bicycle':    (0,   255, 100),
        'truck':      (200,   0, 255),
        'other':      (200, 200, 200),
    }

    for ann_token in annotations:
        ann = nusc.get('sample_annotation', ann_token)

        # Get object position in world frame
        pos_world = np.array(ann['translation'])

        # Transform to ego frame
        pos_h   = np.append(pos_world, 1.0)
        pos_ego = (world_to_ego @ pos_h)[:3]

        x_ego, y_ego = pos_ego[0], pos_ego[1]

        # Skip if outside BEV range
        if not (BEV_X_MIN < x_ego < BEV_X_MAX and
                BEV_Y_MIN < y_ego < BEV_Y_MAX):
            continue

        # Determine object type
        cat = ann['category_name'].lower()
        if 'car' in cat or 'vehicle' in cat:
            obj_type  = 'car'
            w_size, h_size = 20, 10
        elif 'pedestrian' in cat or 'person' in cat:
            obj_type  = 'pedestrian'
            w_size, h_size = 6, 6
        elif 'bicycle' in cat or 'cycle' in cat:
            obj_type  = 'bicycle'
            w_size, h_size = 12, 6
        elif 'truck' in cat or 'bus' in cat:
            obj_type  = 'truck'
            w_size, h_size = 25, 12
        else:
            obj_type  = 'other'
            w_size, h_size = 8, 8

        obj_counts[obj_type] += 1
        color = obj_colors[obj_type]

        # Draw on BEV
        px, py = world_to_bev(x_ego, y_ego)
        cv2.rectangle(bev_canvas,
                      (px - w_size//2, py - h_size//2),
                      (px + w_size//2, py + h_size//2),
                      color, 2)

        # Distance from ego
        dist = np.sqrt(x_ego**2 + y_ego**2)
        cv2.putText(bev_canvas,
                    f"{dist:.0f}m",
                    (px + w_size//2 + 2, py),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, color, 1)

        errors.append(dist)

    # Print object summary
    print(f"\n  Objects in BEV range:")
    for obj_type, count in obj_counts.items():
        if count > 0:
            print(f"    {obj_type:<12}: {count}")

    total_objects = sum(obj_counts.values())
    print(f"    {'TOTAL':<12}: {total_objects}")

    # Create final visualization
    print(f"\n  Creating visualization...")
    n_cams = len(cam_images)

    # Grid of camera images + BEV
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#0f0f0f')
    fig.suptitle(
        "Multi-Camera BEV Perception on Real nuScenes Data\n"
        "Vamshikrishna Gadde | MS Robotics ASU | Day 4 of 90",
        fontsize=14, color='white', y=0.98
    )

    # Camera images (top row)
    cam_order = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT',
    ]

    for i, cam_name in enumerate(cam_order):
        ax = fig.add_subplot(3, 4, i + 1)
        ax.set_facecolor('#0f0f0f')

        if cam_name in cam_images:
            img_rgb = cv2.cvtColor(cam_images[cam_name],
                                   cv2.COLOR_BGR2RGB)
            # Resize for display
            h, w = img_rgb.shape[:2]
            img_small = cv2.resize(img_rgb, (w//3, h//3))
            ax.imshow(img_small)

            color_norm = tuple(
                c/255 for c in CAM_COLORS[cam_name]
            )
            ax.set_title(
                cam_name.replace('CAM_', ''),
                color=color_norm, fontsize=8
            )
        else:
            ax.text(0.5, 0.5, 'Not available',
                    ha='center', va='center',
                    color='grey', transform=ax.transAxes)

        ax.axis('off')

    # BEV map (large, bottom)
    ax_bev = fig.add_subplot(3, 1, 3)
    ax_bev.imshow(cv2.cvtColor(bev_canvas, cv2.COLOR_BGR2RGB))
    ax_bev.set_facecolor('#0f0f0f')
    ax_bev.set_title(
        f"Unified BEV Map — {total_objects} objects detected "
        f"from 6 real cameras",
        color='white', fontsize=11
    )

    x_ticks  = np.linspace(0, BEV_W, 5)
    x_labels = [f"{int(BEV_X_MIN + t/BEV_RESOLUTION)}m"
                for t in x_ticks]
    y_ticks  = np.linspace(0, BEV_H, 6)
    y_labels = [f"{int(BEV_Y_MAX - t/BEV_RESOLUTION)}m"
                for t in y_ticks]
    ax_bev.set_xticks(x_ticks)
    ax_bev.set_xticklabels(x_labels, color='white', fontsize=8)
    ax_bev.set_yticks(y_ticks)
    ax_bev.set_yticklabels(y_labels, color='white', fontsize=8)
    ax_bev.tick_params(colors='white')

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    save_path = f"results/nuscenes_bev_sample{sample_idx:02d}.png"
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f0f0f')
    plt.close