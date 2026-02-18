#!/usr/bin/env python3
"""
Generate a reachability point cloud: all ground-plane positions
where the robot can stand without colliding with the TSDF scene.

Outputs: reachability.ply in the data directory.

Usage:
    python3 generate_reachability.py
"""

import os
import time

import numpy as np
import open3d as o3d

from rrt_point3d import PathPlanner
from path_planner_o3d import (
    PCD_PATH,
    VOXEL_SIZE,
    CROP_VERT,
    Z_PLANE,
    Z_BOUND_EPS,
    BOUND_MARGIN,
    POLY_VERT_BAND,
    POLY_MARGIN,
    ENABLE_FLOOR_CUT,
    FLOOR_CUT,
    pcd_to_invalid_voxels,
    aabb_to_bound,
    build_robot_poly_slices,
)

# Grid resolution for sampling (meters)
GRID_STEP = 0.05


def main():
    # ---- Load TSDF point cloud (already in XY-ground, Z-up frame) ----
    print(f"[1/4] Loading point cloud: {PCD_PATH}")
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Empty point cloud: {PCD_PATH}")

    # ---- Build occupancy voxels ----
    print("[2/4] Building occupancy voxels...")
    vg, invalid_vx_all, pcd_used = pcd_to_invalid_voxels(
        pcd, VOXEL_SIZE, crop_vert=CROP_VERT
    )
    print(f"  Invalid voxels: {invalid_vx_all.shape[0]}")

    if ENABLE_FLOOR_CUT:
        invalid_vx_all = invalid_vx_all[invalid_vx_all[:, 2] > FLOOR_CUT]
        print(f"  After floor cut: {invalid_vx_all.shape[0]}")

    # ---- Bounds ----
    bound = aabb_to_bound(pcd_used.get_axis_aligned_bounding_box(), BOUND_MARGIN)
    bound["low_z"] = float(Z_PLANE - Z_BOUND_EPS)
    bound["high_z"] = float(Z_PLANE + Z_BOUND_EPS)

    # ---- Robot geometry ----
    poly_slices, _ = build_robot_poly_slices()
    print(f"  Robot polygon slices: {len(poly_slices)}")

    # ---- Set up planner (for collision checking only) ----
    print("[3/4] Setting up collision checker...")
    planner = PathPlanner()
    planner.use_state(use_invx=True)
    planner.update_collision_radius(0, VOXEL_SIZE * 2.0)
    planner.update_sp(bound, None, invalid_vx_all, input_vx_size=VOXEL_SIZE)
    planner.set_robot_polygon_stack(
        slices=poly_slices,
        z_mode="offset",
        z_band=POLY_VERT_BAND,
        margin=POLY_MARGIN,
    )
    planner.use_validity_checker("poly_stack")

    # ---- Sample grid on ground plane ----
    xs = np.arange(bound["low_x"], bound["high_x"], GRID_STEP)
    ys = np.arange(bound["low_y"], bound["high_y"], GRID_STEP)
    total = len(xs) * len(ys)
    print(
        f"[4/4] Checking {len(xs)} x {len(ys)} = {total} positions "
        f"(step={GRID_STEP}m)..."
    )

    reachable = []
    checked = 0
    t0 = time.time()

    for i, x in enumerate(xs):
        for y in ys:
            state = np.array([x, y, Z_PLANE])
            if planner.isStateValid_poly_stack(state):
                reachable.append(state)
        checked += len(ys)
        if (i + 1) % 20 == 0 or (i + 1) == len(xs):
            elapsed = time.time() - t0
            pct = 100.0 * checked / total
            print(f"  {pct:5.1f}%  ({checked}/{total})  "
                  f"reachable so far: {len(reachable)}  "
                  f"elapsed: {elapsed:.1f}s")

    dt = time.time() - t0
    print(f"\nDone: {len(reachable)} reachable / {total} total "
          f"({100.0 * len(reachable) / total:.1f}%) in {dt:.1f}s")

    if len(reachable) == 0:
        print("No reachable points found. Check robot geometry / margins.")
        return

    # ---- Ground filter: keep only points with ±Z scene neighbours ----
    print("Filtering for ground (±Z neighbour check)...")
    scene_pts = np.asarray(pcd.points)
    scene_vx = np.round(scene_pts / VOXEL_SIZE).astype(np.int64)
    scene_vx_set = set(map(tuple, scene_vx))

    reachable_arr = np.array(reachable)
    reach_vx = np.round(reachable_arr / VOXEL_SIZE).astype(np.int64)

    z_search = int(round(0.5 / VOXEL_SIZE))  # ±0.5 m in voxel steps
    offsets_z = range(-z_search, z_search + 1)

    ground_mask = np.array([
        any(tuple(v + [0, 0, dz]) in scene_vx_set for dz in offsets_z if dz != 0)
        for v in reach_vx
    ])

    n_before = len(reachable)
    reachable = list(reachable_arr[ground_mask])
    print(f"  Ground filter: {len(reachable)} / {n_before} kept")

    if len(reachable) == 0:
        print("No ground-contact points remain.")
        return

    # ---- Keep only the largest cluster ----
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(np.array(reachable))
    labels = np.array(pcd_tmp.cluster_dbscan(eps=GRID_STEP * 2, min_points=3))
    if labels.max() >= 0:
        largest = np.argmax(np.bincount(labels[labels >= 0]))
        reachable = [r for r, l in zip(reachable, labels) if l == largest]
        print(f"  Largest cluster: {len(reachable)} pts (of {int((labels >= 0).sum())} clustered, {int((labels == -1).sum())} noise)")

    # ---- Save (already in XY-ground, Z-up frame) ----
    pcd_reach = o3d.geometry.PointCloud()
    pcd_reach.points = o3d.utility.Vector3dVector(np.array(reachable))
    pcd_reach.paint_uniform_color([0.0, 1.0, 0.0])  # green

    out_path = os.path.join(os.path.dirname(PCD_PATH), "reachability.ply")
    o3d.io.write_point_cloud(out_path, pcd_reach)
    print(f"Saved to {out_path}")

    # ---- Visualize: TSDF (original colors) + reachability (green) ----
    pcd_scene = o3d.io.read_point_cloud(PCD_PATH)
    pcd_scene = pcd_scene.voxel_down_sample(voxel_size=0.02)

    o3d.visualization.draw_geometries(
        [pcd_scene, pcd_reach],
        window_name="Reachability (green) + Scene",
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
