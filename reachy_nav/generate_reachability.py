#!/usr/bin/env python3
"""
Generate a reachability point cloud: all ground-plane positions
where the robot can stand without colliding with the TSDF scene.

Outputs: reachability.ply next to the input PLY.

Usage:
    python3 generate_reachability.py <path/to/tsdf_fused.ply>
"""

import os
import sys
import time

import numpy as np
import open3d as o3d

from rrt_point3d import PathPlanner

# ----------------------------
# Config
# ----------------------------
VOXEL_SIZE = 0.1
BOUND_MARGIN = 0.5
CROP_VERT = None

Z_BOUND_EPS = 1e-3
Z_PLANE_OFFSET = 0.35 # raise sampling plane above ground to avoid ground collision

ENABLE_FLOOR_CUT = False
FLOOR_CUT_OFFSET = 0.03  # added to Z_PLANE when floor cut is enabled

# Robot geometry
CYL_Z_REL0 = -1.00
CYL_Z_REL1 = -1.20
CYL_RADIUS = 0.35

RECT_Z_REL0 = 0.10
RECT_Z_REL1 = -1.00
RECT_WIDTH_X = 0.30
RECT_THICK_Y = 0.60

TORSO_ABOVE_PLANE = 1.0
VERT_STEP = 0.10

POLY_VERT_BAND = 0.30
POLY_MARGIN = 0.00

GRID_STEP = 0.1

# ----------------------------
# Ground plane estimation
# ----------------------------
def estimate_ground_z(pcd, bin_size=0.05, gap_threshold=0.15):
    """Estimate ground-plane Z by 1D height clustering.

    1. Histogram all Z values into small bins.
    2. Group contiguous non-empty bins into clusters (split when gap > threshold).
    3. Return the weighted-average Z of the largest cluster (most points).
    """
    zs = np.asarray(pcd.points)[:, 2]
    bins = np.arange(zs.min(), zs.max() + bin_size, bin_size)
    counts, edges = np.histogram(zs, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # Keep only non-empty bins
    nonzero = counts > 0
    if not np.any(nonzero):
        raise RuntimeError("Point cloud has no points for ground estimation")
    nz_centers = centers[nonzero]
    nz_counts = counts[nonzero]

    # Group contiguous bins into clusters
    clusters = []
    cur_c = [nz_centers[0]]
    cur_w = [nz_counts[0]]
    for i in range(1, len(nz_centers)):
        if nz_centers[i] - nz_centers[i - 1] > gap_threshold:
            clusters.append((cur_c, cur_w))
            cur_c, cur_w = [], []
        cur_c.append(nz_centers[i])
        cur_w.append(nz_counts[i])
    clusters.append((cur_c, cur_w))

    # Largest cluster by total point count
    best_c, best_w = max(clusters, key=lambda c: sum(c[1]))
    ground_z = float(np.average(best_c, weights=best_w))
    return ground_z


# ----------------------------
# Helpers (inlined from path_planner_o3d)
# ----------------------------
def pcd_to_invalid_voxels(pcd, voxel_size, crop_vert=None):
    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty.")
    if crop_vert is not None:
        zmin, zmax = crop_vert
        pts = np.asarray(pcd.points)
        mask = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        pcd = pcd.select_by_index(np.where(mask)[0])
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size * 0.5)
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    voxels = vg.get_voxels()
    if len(voxels) == 0:
        raise ValueError("VoxelGrid is empty after voxelization.")
    centers = np.array([vg.get_voxel_center_coordinate(v.grid_index) for v in voxels], dtype=np.float64)
    return vg, centers, pcd


def aabb_to_bound(aabb, margin):
    mn = aabb.get_min_bound()
    mx = aabb.get_max_bound()
    return {
        "low_x": float(mn[0] - margin), "high_x": float(mx[0] + margin),
        "low_y": float(mn[1] - margin), "high_y": float(mx[1] + margin),
        "low_z": float(mn[2] - margin), "high_z": float(mx[2] + margin),
    }


def sample_z_range(z0, z1, step):
    if z1 < z0:
        z0, z1 = z1, z0
    zs = []
    z = z0
    while z <= z1 + 1e-9:
        zs.append(float(np.round(z, 6)))
        z += step
    if len(zs) == 0 or abs(zs[-1] - z1) > 1e-6:
        zs.append(float(np.round(z1, 6)))
    return zs


def make_circle_polygon(radius, n=32):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)


def make_rectangle_polygon(width_x, thick_y):
    hx = width_x * 0.5
    hy = thick_y * 0.5
    return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)


def build_robot_poly_slices():
    poly_cyl = make_circle_polygon(CYL_RADIUS, n=40)
    poly_rect = make_rectangle_polygon(RECT_WIDTH_X, RECT_THICK_Y)
    cyl_off0 = TORSO_ABOVE_PLANE + CYL_Z_REL0
    cyl_off1 = TORSO_ABOVE_PLANE + CYL_Z_REL1
    rect_off0 = TORSO_ABOVE_PLANE + RECT_Z_REL0
    rect_off1 = TORSO_ABOVE_PLANE + RECT_Z_REL1
    cyl_zs = sample_z_range(cyl_off0, cyl_off1, VERT_STEP)
    rect_zs = sample_z_range(rect_off0, rect_off1, VERT_STEP)
    slices = [(z, poly_cyl) for z in cyl_zs] + [(z, poly_rect) for z in rect_zs]
    return slices, (cyl_off0, cyl_off1, rect_off0, rect_off1)


# ----------------------------
# Main
# ----------------------------
def main(pcd_path: str):
    # ---- Load TSDF point cloud ----
    print(f"[1/5] Loading point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Empty point cloud: {pcd_path}")

    # ---- Estimate ground plane height ----
    print("[2/5] Estimating ground plane height...")
    z_ground = estimate_ground_z(pcd)
    z_plane = z_ground + Z_PLANE_OFFSET
    print(f"  Estimated ground Z: {z_ground:.4f}")
    print(f"  Sampling plane Z:   {z_plane:.4f}  (ground + {Z_PLANE_OFFSET}m)")

    floor_cut = z_ground + FLOOR_CUT_OFFSET

    # ---- Build occupancy voxels ----
    print("[3/5] Building occupancy voxels...")
    vg, invalid_vx_all, pcd_used = pcd_to_invalid_voxels(
        pcd, VOXEL_SIZE, crop_vert=CROP_VERT
    )
    print(f"  Invalid voxels: {invalid_vx_all.shape[0]}")

    if ENABLE_FLOOR_CUT:
        invalid_vx_all = invalid_vx_all[invalid_vx_all[:, 2] > floor_cut]
        print(f"  After floor cut (z > {floor_cut:.3f}): {invalid_vx_all.shape[0]}")

    # ---- Bounds ----
    bound = aabb_to_bound(pcd_used.get_axis_aligned_bounding_box(), BOUND_MARGIN)
    bound["low_z"] = float(z_plane - Z_BOUND_EPS)
    bound["high_z"] = float(z_plane + Z_BOUND_EPS)

    # ---- Robot geometry ----
    poly_slices, _ = build_robot_poly_slices()
    print(f"  Robot polygon slices: {len(poly_slices)}")

    # ---- Set up planner (for collision checking only) ----
    print("[4/5] Setting up collision checker...")
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
        f"[5/5] Checking {len(xs)} x {len(ys)} = {total} positions "
        f"(step={GRID_STEP}m, z_plane={z_plane:.4f})..."
    )

    reachable = []
    checked = 0
    t0 = time.time()

    for i, x in enumerate(xs):
        for y in ys:
            state = np.array([x, y, z_plane])
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

    # ---- Ground support filter: keep only points with floor beneath ----
    from scipy.spatial import cKDTree

    ground_tolerance = VOXEL_SIZE * 3.0  # vertical band around z_ground
    search_radius = VOXEL_SIZE * 2.0

    scene_pts = np.asarray(pcd.points)
    ground_mask_scene = np.abs(scene_pts[:, 2] - z_ground) <= ground_tolerance
    ground_xy = scene_pts[ground_mask_scene, :2]

    print(f"Ground support filter: {ground_xy.shape[0]} scene points near "
          f"z_ground={z_ground:.3f} (±{ground_tolerance:.3f}m)")

    if ground_xy.shape[0] > 0:
        tree = cKDTree(ground_xy)
        reachable_arr = np.array(reachable)
        dists, _ = tree.query(reachable_arr[:, :2])
        supported = dists <= search_radius

        n_before = len(reachable)
        reachable = list(reachable_arr[supported])
        print(f"  Ground support: {len(reachable)} / {n_before} kept "
              f"(radius={search_radius:.3f}m)")
    else:
        print("  WARNING: no ground-level scene points found, skipping filter")

    if len(reachable) == 0:
        print("No ground-supported points remain.")
        return

    # ---- Keep only the largest cluster ----
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(np.array(reachable))
    labels = np.array(pcd_tmp.cluster_dbscan(eps=GRID_STEP * 2, min_points=3))
    if labels.max() >= 0:
        largest = np.argmax(np.bincount(labels[labels >= 0]))
        reachable = [r for r, l in zip(reachable, labels) if l == largest]
        print(f"  Largest cluster: {len(reachable)} pts (of {int((labels >= 0).sum())} clustered, {int((labels == -1).sum())} noise)")

    # ---- Save ----
    pcd_reach = o3d.geometry.PointCloud()
    pcd_reach.points = o3d.utility.Vector3dVector(np.array(reachable))
    pcd_reach.paint_uniform_color([0.0, 1.0, 0.0])

    out_path = os.path.join(os.path.dirname(pcd_path), "reachability.ply")
    o3d.io.write_point_cloud(out_path, pcd_reach)
    print(f"Saved to {out_path}")

    # ---- Visualize ----
    pcd_scene = o3d.io.read_point_cloud(pcd_path)
    pcd_scene = pcd_scene.voxel_down_sample(voxel_size=0.02)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    o3d.visualization.draw_geometries(
        [pcd_scene, pcd_reach, origin],
        window_name="Reachability (green) + Scene",
        point_show_normal=False,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <path/to/tsdf_fused.ply>")
        sys.exit(1)
    main(sys.argv[1])
