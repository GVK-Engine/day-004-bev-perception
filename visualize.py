"""
visualize.py
============
Advanced BEV visualization tools.

What this does:
  Creates professional multi-panel visualizations
  Shows all 6 camera images alongside BEV map
  Draws object bounding boxes with labels
  Creates comparison: camera view vs BEV view
  Saves high-quality output for portfolio/LinkedIn

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
import matplotlib.gridspec as gridspec
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# ── SETTINGS ──────────────────────────────────────────────────────────

NUSCENES_ROOT  = r"D:\day-004-bev-perception"
VERSION        = "v1.0-mini"
RESULTS_DIR    = "results"

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
    'CAM_FRONT':       '#FFC800',
    'CAM_FRONT_RIGHT': '#00C8FF',
    'CAM_FRONT_LEFT':  '#00FF64',
    'CAM_BACK':        '#FF6400',
    'CAM_BACK_RIGHT':  '#C800FF',
    'CAM_BACK_LEFT':   '#FF0064',
}

OBJ_COLORS = {
    'car':        (0,   200, 255),
    'pedestrian': (255, 100,   0),
    'bicycle':    (0,   255, 100),
    'truck':      (200,   0, 255),
    'motorcycle': (255,   0, 200),
    'other':      (200, 200, 200),
}


# ── HELPERS ───────────────────────────────────────────────────────────

def get_transform(record):
    T = np.eye(4)
    T[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    T[:3,  3] = np.array(record['translation'])
    return T


def world_to_bev(x, y):
    px = int((x - BEV_X_MIN) * BEV_RESOLUTION)
    py = int(BEV_H - (y - BEV_Y_MIN) * BEV_RESOLUTION)
    return px, py


def get_obj_type(cat):
    cat = cat.lower()
    if 'car' in cat:        return 'car'
    if 'pedestrian' in cat: return 'pedestrian'
    if 'bicycle' in cat:    return 'bicycle'
    if 'truck' in cat:      return 'truck'
    if 'motorcycle' in cat: return 'motorcycle'
    return 'other'


def project_to_bev(image, intrinsic, cam_to_ego):
    """Project camera image onto BEV grid using IPM."""
    bev       = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    img_h, img_w = image.shape[:2]
    ego_to_cam   = np.linalg.inv(cam_to_ego)

    px_arr  = np.arange(BEV_W)
    py_arr  = np.arange(BEV_H)
    px_grid, py_grid = np.meshgrid(px_arr, py_arr)

    x_world = px_grid / BEV_RESOLUTION + BEV_X_MIN
    y_world = (BEV_H - py_grid) / BEV_RESOLUTION + BEV_Y_MIN

    N       = BEV_W * BEV_H
    ego_pts = np.ones((4, N), dtype=np.float64)
    ego_pts[0] = x_world.flatten()
    ego_pts[1] = y_world.flatten()
    ego_pts[2] = 0.0

    cam_pts = ego_to_cam @ ego_pts
    valid   = cam_pts[2] > 0.5

    K    = np.array(intrinsic)
    proj = K @ cam_pts[:3]
    u    = proj[0] / proj[2]
    v    = proj[1] / proj[2]

    valid &= (u >= 0) & (u < img_w-1) & \
             (v >= 0) & (v < img_h-1)

    u_v      = u[valid].astype(int)
    v_v      = v[valid].astype(int)
    colors   = image[v_v, u_v]
    flat_idx = np.arange(N)[valid]
    bev[flat_idx // BEV_W, flat_idx % BEV_W] = colors

    return bev


# ── MAIN VISUALIZATION ────────────────────────────────────────────────

def create_full_visualization(sample_idx=0):
    """
    Create a complete professional visualization for one sample.

    Layout:
      Row 1: CAM_FRONT_LEFT | CAM_FRONT | CAM_FRONT_RIGHT
      Row 2: CAM_BACK_LEFT  | CAM_BACK  | CAM_BACK_RIGHT
      Row 3: Full-width BEV map with objects and labels
    """
    print(f"\n  Creating visualization for sample {sample_idx}...")

    nusc = NuScenes(
        version  = VERSION,
        dataroot = NUSCENES_ROOT,
        verbose  = False
    )

    sample = nusc.sample[sample_idx]

    # Build BEV canvas
    bev = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    bev[:] = (25, 25, 25)

    # Road surface
    cx = BEV_W // 2
    cv2.rectangle(bev, (cx-22, 0), (cx+22, BEV_H),
                  (55, 55, 55), -1)

    # Lane markings
    for y in range(0, BEV_H, 40):
        cv2.rectangle(bev, (cx-2, y),
                      (cx+2, y+20), (170, 170, 70), -1)

    cam_images = {}

    # Load and project all cameras
    print(f"  Projecting 6 cameras...")
    for cam_name in CAMERAS:
        if cam_name not in sample['data']:
            continue

        cam_token  = sample['data'][cam_name]
        cam_data   = nusc.get('sample_data', cam_token)
        img_path   = os.path.join(NUSCENES_ROOT, cam_data['filename'])

        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        cam_images[cam_name] = image

        calib_token = cam_data['calibrated_sensor_token']
        calib       = nusc.get('calibrated_sensor', calib_token)
        cam_to_ego  = get_transform(calib)
        intrinsic   = calib['camera_intrinsic']

        cam_bev = project_to_bev(image, intrinsic, cam_to_ego)

        mask = cam_bev.sum(axis=2) > 0
        bev[mask] = (bev[mask] * 0.25 +
                     cam_bev[mask] * 0.75).astype(np.uint8)

    # Distance rings
    ego_px, ego_py = world_to_bev(0, 0)
    for dist in [10, 20, 30, 40]:
        _, py_ring = world_to_bev(0, dist)
        radius = abs(ego_py - py_ring)
        cv2.circle(bev, (ego_px, ego_py),
                   radius, (70, 70, 70), 1)
        cv2.putText(bev, f'{dist}m',
                    (ego_px + 5, py_ring),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (110, 110, 110), 1)

    # Ego vehicle
    cv2.rectangle(bev,
                  (ego_px-8, ego_py-18),
                  (ego_px+8, ego_py+4),
                  (255, 255, 255), -1)
    cv2.putText(bev, 'EGO',
                (ego_px-12, ego_py+16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 220, 0), 1)

    # Get ego pose
    cam_token    = sample['data']['CAM_FRONT']
    cam_data     = nusc.get('sample_data', cam_token)
    ego_token    = cam_data['ego_pose_token']
    ego_pose     = nusc.get('ego_pose', ego_token)
    ego_to_world = get_transform(ego_pose)
    world_to_ego = np.linalg.inv(ego_to_world)

    # Draw objects
    obj_counts = {}
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        pos_world = np.array(ann['translation'])
        pos_h     = np.append(pos_world, 1.0)
        pos_ego   = (world_to_ego @ pos_h)[:3]

        x, y = pos_ego[0], pos_ego[1]
        if not (BEV_X_MIN < x < BEV_X_MAX and
                BEV_Y_MIN < y < BEV_Y_MAX):
            continue

        obj_type = get_obj_type(ann['category_name'])
        color    = OBJ_COLORS.get(obj_type, (200, 200, 200))
        obj_counts[obj_type] = obj_counts.get(obj_type, 0) + 1

        size = ann['size']
        w_px = max(int(size[0] * BEV_RESOLUTION), 8)
        h_px = max(int(size[1] * BEV_RESOLUTION), 8)

        px, py = world_to_bev(x, y)
        dist   = np.sqrt(x**2 + y**2)

        cv2.rectangle(bev,
                      (px-w_px//2, py-h_px//2),
                      (px+w_px//2, py+h_px//2),
                      color, 2)
        cv2.putText(bev,
                    f"{obj_type[0].upper()} {dist:.0f}m",
                    (px-w_px//2, py-h_px//2-3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, color, 1)

    total = sum(obj_counts.values())
    print(f"  Objects in BEV: {total}")

    # Build figure
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0d0d0d')

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.08,
        wspace=0.05,
        height_ratios=[1, 1, 1.4]
    )

    fig.suptitle(
        "Multi-Camera BEV Perception — Real nuScenes Data\n"
        "Vamshikrishna Gadde | MS Robotics ASU | Day 4 of 90",
        fontsize=14, color='white', y=0.99
    )

    # Camera grid
    cam_layout = [
        [('CAM_FRONT_LEFT',  0, 0),
         ('CAM_FRONT',       0, 1),
         ('CAM_FRONT_RIGHT', 0, 2)],
        [('CAM_BACK_LEFT',   1, 0),
         ('CAM_BACK',        1, 1),
         ('CAM_BACK_RIGHT',  1, 2)],
    ]

    for row in cam_layout:
        for cam_name, r, c in row:
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor('#0d0d0d')

            if cam_name in cam_images:
                img_rgb = cv2.cvtColor(
                    cam_images[cam_name], cv2.COLOR_BGR2RGB
                )
                h, w = img_rgb.shape[:2]
                img_s = cv2.resize(img_rgb, (w//3, h//3))
                ax.imshow(img_s)

            hex_color = CAM_COLORS.get(cam_name, '#FFFFFF')
            ax.set_title(
                cam_name.replace('CAM_', ''),
                color=hex_color, fontsize=9,
                fontweight='bold', pad=3
            )
            ax.axis('off')

    # BEV map — full width bottom
    ax_bev = fig.add_subplot(gs[2, :])
    ax_bev.set_facecolor('#0d0d0d')
    ax_bev.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))

    title_parts = [f"{k}: {v}" for k, v in obj_counts.items()]
    ax_bev.set_title(
        f"Unified BEV — {total} objects detected  |  "
        + "  ".join(title_parts),
        color='white', fontsize=11, pad=6
    )

    x_ticks  = np.linspace(0, BEV_W, 5)
    x_labels = [f"{int(BEV_X_MIN + t/BEV_RESOLUTION)}m"
                for t in x_ticks]
    y_ticks  = np.linspace(0, BEV_H, 6)
    y_labels = [f"{int(BEV_Y_MAX - t/BEV_RESOLUTION)}m"
                for t in y_ticks]

    ax_bev.set_xticks(x_ticks)
    ax_bev.set_xticklabels(x_labels, color='white', fontsize=9)
    ax_bev.set_yticks(y_ticks)
    ax_bev.set_yticklabels(y_labels, color='white', fontsize=9)
    ax_bev.tick_params(colors='white')

    for spine in ax_bev.spines.values():
        spine.set_edgecolor('#444444')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(
        RESULTS_DIR, f"bev_full_sample{sample_idx:02d}.png"
    )
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0d0d0d')
    plt.close()
    print(f"  Saved: {path}")
    return path


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BEV Visualization Tool")
    print("="*60)

    # Create visualizations for first 3 samples
    for i in range(3):
        create_full_visualization(sample_idx=i)

    print("\n  All visualizations saved to results/")
    print("  Open results/bev_full_sample00.png to see!")