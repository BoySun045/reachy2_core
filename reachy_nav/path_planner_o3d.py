import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as Rot, Slerp
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from rrt_point3d import PathPlanner


# ----------------------------
# Helpers (visualization)
# ----------------------------
def voxelgrid_to_lineset(voxel_grid: o3d.geometry.VoxelGrid) -> o3d.geometry.LineSet:
    """Convert an Open3D VoxelGrid into a wireframe LineSet (one cube per voxel)."""
    voxels = voxel_grid.get_voxels()
    vs = voxel_grid.voxel_size
    # 8 corners of a unit cube, scaled by voxel_size
    offsets = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64) * vs
    # 12 edges of a cube
    cube_edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]

    all_pts = []
    all_lines = []
    for v in voxels:
        center = voxel_grid.get_voxel_center_coordinate(v.grid_index)
        origin = center - 0.5 * vs
        base = len(all_pts)
        all_pts.extend((origin + offsets).tolist())
        all_lines.extend([[base + a, base + b] for a, b in cube_edges])

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_pts),
        lines=o3d.utility.Vector2iVector(all_lines),
    )
    return ls


# ----------------------------
# User config
# ----------------------------
# PLY frame: XY = ground plane, Z+ = up.
# All config values below are in this frame.

PCD_PATH = os.path.join(THIS_DIR, "data_dso/2026_02_16-12_02_34-default_experiment/tsdf_fused.ply")

VOXEL_SIZE = 0.10
BOUND_MARGIN = 0.5

# Optional crop on vertical axis (Z). Set to None to disable.
CROP_VERT = None  # e.g. (0.0, 2.0) in Z values

# ---- Planar motion constraints ----
Z_PLANE = 0.58      # ground-plane height (Z+ is up); tune to your map
Z_BOUND_EPS = 1e-3  # planner bounds tightness on vertical axis

# Start / Goal in [X, Y, Z] — Z component is overridden to Z_PLANE
START_POS = np.array([0.0, 0.0, 0.0], dtype=float)
START_YAW = np.pi/2        # radians, robot X-axis direction at start
GOAL_POS  = np.array([1.5, -5.0, 0.0], dtype=float)
GOAL_YAW  = None       # None = face object/last segment; or set radians

# Planner
TIME_LIMIT = 5.0
METHOD = "rrtstar"

# ----------------------------
# Robot geometry (torso-relative spec)
# ----------------------------
# Z+ is up, so "above torso" = positive offset, "below torso" = negative offset
# - cylinder (base) between -1.00 and -1.20 m from torso (at/below ground), radius 0.25 m
# - rectangle (body) from +0.10 (above torso) to -1.00 m (ground level), 0.50 x 0.20

CYL_Z_REL0 = -1.00
CYL_Z_REL1 = -1.20
CYL_RADIUS = 0.25

RECT_Z_REL0 = 0.10
RECT_Z_REL1 = -1.00
RECT_WIDTH_X = 0.20
RECT_THICK_Y = 0.50

# Torso is above ground → positive Z offset from Z_PLANE
TORSO_ABOVE_PLANE = 1.0

# Discretization along vertical (more slices = safer but slower)
VERT_STEP = 0.10  # try 0.15 if too slow

# Collision checker params
POLY_VERT_BAND = 0.30   # obstacle slab thickness per slice
POLY_MARGIN = 0.00      # safety margin around polygon edges 0.05

# OPTIONAL: remove floor voxels
ENABLE_FLOOR_CUT = False
FLOOR_CUT = Z_PLANE + 0.03  # keep only voxels with Z > this

# Visualization
SHOW_ROBOT_POLYS_AT_START_GOAL = True
SHOW_ROBOT_POLYS_ALONG_PATH = True
ROBOT_POLY_SUBSAMPLE = 4
MAX_POLY_SLICES_TO_DRAW = 40

# Debug visualization for invalid start/goal
SHOW_COLLISION_POINTS = True
MAX_COLLISION_POINTS_PER_SLICE = 300


# ----------------------------
# Helpers
# ----------------------------
def pcd_to_invalid_voxels(pcd: o3d.geometry.PointCloud, voxel_size: float, crop_vert=None):
    """crop_vert: (min, max) range on vertical Z axis."""
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
        raise ValueError("VoxelGrid is empty after voxelization (try smaller VOXEL_SIZE or check cloud).")

    centers = np.array([vg.get_voxel_center_coordinate(v.grid_index) for v in voxels], dtype=np.float64)
    return vg, centers, pcd


def aabb_to_bound(aabb: o3d.geometry.AxisAlignedBoundingBox, margin: float):
    mn = aabb.get_min_bound()
    mx = aabb.get_max_bound()
    return {
        "low_x": float(mn[0] - margin),
        "high_x": float(mx[0] + margin),
        "low_y": float(mn[1] - margin),
        "high_y": float(mx[1] + margin),
        "low_z": float(mn[2] - margin),
        "high_z": float(mx[2] + margin),
    }


def force_z(p: np.ndarray, z_plane: float) -> np.ndarray:
    q = np.array(p, dtype=float).copy()
    q[2] = z_plane
    return q


def sample_z_range(z0: float, z1: float, step: float):
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


def make_circle_polygon(radius: float, n: int = 32) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)


def make_rectangle_polygon(width_x: float, thick_y: float) -> np.ndarray:
    hx = width_x * 0.5
    hy = thick_y * 0.5
    return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)


def make_poly_lineset(poly_world_xy: np.ndarray, z: float, color=(1.0, 0.0, 0.0)):
    pts3 = np.column_stack([poly_world_xy, np.full((poly_world_xy.shape[0],), z, dtype=float)])
    lines = [[i, (i + 1) % len(pts3)] for i in range(len(pts3))]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts3),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=float), (len(lines), 1)))
    return ls


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Yaw angle (radians, around Z) → quaternion [x, y, z, w]."""
    return Rot.from_euler("z", yaw).as_quat()


def quat_to_yaw(quat) -> float:
    """Quaternion [x, y, z, w] → yaw angle (radians)."""
    return Rot.from_quat(quat).as_euler("xyz")[2]


def assign_look_at_orientations(solution, start_yaw, goal_yaw=None):
    """Set quaternions along a path.

    - First waypoint keeps *start_yaw*.
    - Intermediate waypoints (1..n-2) face the next waypoint.
    - Last waypoint keeps *goal_yaw* (or faces along the last segment
      if *goal_yaw* is None).
    """
    n = len(solution)
    if n == 0:
        return solution

    # First waypoint: always start_yaw
    solution[0]["quat"] = yaw_to_quat(start_yaw)

    if n == 1:
        return solution

    # Intermediate waypoints: face the next one
    for i in range(1, n - 1):
        curr = solution[i]["pos"]
        nxt = solution[i + 1]["pos"]
        dx = nxt[0] - curr[0]
        dy = nxt[1] - curr[1]
        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            yaw = np.arctan2(dy, dx)
        else:
            yaw = quat_to_yaw(solution[i - 1]["quat"])
        solution[i]["quat"] = yaw_to_quat(yaw)

    # Last waypoint
    if goal_yaw is not None:
        solution[-1]["quat"] = yaw_to_quat(goal_yaw)
    else:
        prev = solution[-2]["pos"]
        curr = solution[-1]["pos"]
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            solution[-1]["quat"] = yaw_to_quat(np.arctan2(dy, dx))
        else:
            solution[-1]["quat"] = solution[-2]["quat"].copy()

    return solution


def build_robot_poly_slices():
    """
    Build (vert_offset, poly_xy_local) slices for z_mode="offset":
      vert_abs = state_vert + vert_offset
    We convert torso-relative Z ranges to offsets using TORSO_ABOVE_PLANE.
    """
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


def debug_state_poly_collisions(planner, state_xyz, poly_slices, max_hits_per_slice=300):
    """
    Returns:
      bad_slices: list of dicts with offending slice info
      coll_pts: (K,3) points that are likely causing collision, for visualization
    """
    if not hasattr(planner, "_invx_pts") or not hasattr(planner, "_invx_z"):
        if hasattr(planner, "_cache_invx_points"):
            planner._cache_invx_points()
        else:
            raise RuntimeError("Planner doesn't have cached invx points. Did you call set_robot_polygon_stack after update_sp?")

    x, y, z_state = float(state_xyz[0]), float(state_xyz[1]), float(state_xyz[2])
    half = 0.5 * planner._poly_z_band
    margin = planner._poly_margin

    invx_pts = planner._invx_pts
    invx_z = planner._invx_z

    bad_slices = []
    all_coll = []

    for (z_off, poly_local) in poly_slices:
        z_abs = z_state + z_off  # z_mode="offset"

        zmask = (invx_z >= (z_abs - half)) & (invx_z <= (z_abs + half))
        cand = invx_pts[zmask]
        if cand.shape[0] == 0:
            continue

        poly_world = poly_local + np.array([x, y], dtype=np.float64)

        # AABB filter
        mn = np.min(poly_world, axis=0) - margin
        mx = np.max(poly_world, axis=0) + margin
        pts_xy = cand[:, :2]
        aabb = (pts_xy[:, 0] >= mn[0]) & (pts_xy[:, 0] <= mx[0]) & \
               (pts_xy[:, 1] >= mn[1]) & (pts_xy[:, 1] <= mx[1])
        cand2 = cand[aabb]
        if cand2.shape[0] == 0:
            continue

        # collision test using planner's own poly logic
        if planner._poly_collides_points(cand2[:, :2], poly_world, margin):
            bad_slices.append({
                "z_off": float(z_off),
                "z_abs": float(z_abs),
                "candidates_after_aabb": int(cand2.shape[0]),
            })
            all_coll.append(cand2[:max_hits_per_slice])

    coll_pts = np.concatenate(all_coll, axis=0) if len(all_coll) else np.zeros((0, 3), dtype=np.float64)
    return bad_slices, coll_pts


def main():
    # ---- load point cloud (already in XY-ground, Z-up frame) ----
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Loaded empty point cloud: {PCD_PATH}")

    # ---- build occupancy voxels from point cloud ----
    vg, invalid_vx_all, pcd_used = pcd_to_invalid_voxels(pcd, VOXEL_SIZE, crop_vert=CROP_VERT)
    print("[OCC] invalid voxels (all):", invalid_vx_all.shape)

    if ENABLE_FLOOR_CUT:
        invalid_vx_all = invalid_vx_all[invalid_vx_all[:, 2] > FLOOR_CUT]
        print("[OCC] after floor cut:", invalid_vx_all.shape, "(cut vert >", FLOOR_CUT, ")")

    # ---- bounds from cloud ----
    bound = aabb_to_bound(pcd_used.get_axis_aligned_bounding_box(), BOUND_MARGIN)

    # ---- restrict vertical bounds to a thin slice around ground plane ----
    bound["low_z"] = float(Z_PLANE - Z_BOUND_EPS)
    bound["high_z"] = float(Z_PLANE + Z_BOUND_EPS)
    print("[BOUND]", bound)

    # ---- force start/goal vertical to ground plane ----
    start_xz = force_z(START_POS, Z_PLANE)
    goal_xz = force_z(GOAL_POS, Z_PLANE)

    # ---- robot slices ----
    poly_slices, (cyl_off0, cyl_off1, rect_off0, rect_off1) = build_robot_poly_slices()
    print(f"[ROBOT] polygon slices = {len(poly_slices)} (VERT_STEP={VERT_STEP})")
    print(f"        torso_above_plane = {TORSO_ABOVE_PLANE}")
    print(f"        cylinder z_rel [{CYL_Z_REL0}, {CYL_Z_REL1}] -> offsets [{cyl_off0:.2f}, {cyl_off1:.2f}]")
    print(f"        rect     z_rel [{RECT_Z_REL0}, {RECT_Z_REL1}] -> offsets [{rect_off0:.2f}, {rect_off1:.2f}]")

    # ---- planner ----
    planner = PathPlanner()
    planner.use_state(use_invx=True)
    planner.update_collision_radius(0, VOXEL_SIZE * 2.0)

    # IMPORTANT: pass FULL 3D invalid voxels
    planner.update_sp(bound, None, invalid_vx_all, input_vx_size=VOXEL_SIZE)

    # polygon stack checker
    planner.set_robot_polygon_stack(
        slices=poly_slices,
        z_mode="offset",
        z_band=POLY_VERT_BAND,
        margin=POLY_MARGIN,
    )
    planner.use_validity_checker("poly_stack")

    print("[DEBUG] start_xz passed:", start_xz)
    print("[DEBUG] goal_xz  passed:", goal_xz)

    # ---- quick validity check BEFORE solve ----
    start_ok = planner.isStateValid_poly_stack(start_xz)
    goal_ok = planner.isStateValid_poly_stack(goal_xz)
    print(f"[CHECK] start valid? {start_ok}  goal valid? {goal_ok}")

    bad_start, coll_start = debug_state_poly_collisions(
        planner, start_xz, poly_slices, max_hits_per_slice=MAX_COLLISION_POINTS_PER_SLICE
    )
    bad_goal, coll_goal = debug_state_poly_collisions(
        planner, goal_xz, poly_slices, max_hits_per_slice=MAX_COLLISION_POINTS_PER_SLICE
    )
    print("[DEBUG] start bad slices (first 10):", bad_start[:10], ("...(more)" if len(bad_start) > 10 else ""))
    print("[DEBUG] goal  bad slices (first 10):", bad_goal[:10], ("...(more)" if len(bad_goal) > 10 else ""))
    print("[DEBUG] collision points start:", coll_start.shape, " goal:", coll_goal.shape)

    start = {"pos": start_xz, "quat": yaw_to_quat(START_YAW)}
    goal  = {"pos": goal_xz,  "quat": yaw_to_quat(GOAL_YAW if GOAL_YAW is not None else 0.0)}
    planner.update_start_goal(start, goal)

    planner.solve(time_limit=TIME_LIMIT, method=METHOD)
    solution = planner.get_solution()


    if solution is None:
        print("[PLANNER] No solution returned.")
        solution = []

    # Clamp solution Z & assign look-at orientations
    for sol in solution:
        sol["pos"][2] = Z_PLANE
    solution = assign_look_at_orientations(solution, START_YAW, GOAL_YAW)

    print("\n[PATH]")
    for i, s in enumerate(solution):
        yaw_deg = np.degrees(quat_to_yaw(s["quat"]))
        print(f"{i}: pos = {s['pos']}  yaw = {yaw_deg:.1f}°")

    # ---- visualization ----
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Origin axes: red=X, green=Y, blue=Z
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(origin_frame)

    vis.add_geometry(voxelgrid_to_lineset(vg))
    vis.add_geometry(pcd_used)

    # Z plane rectangle
    aabb = pcd_used.get_axis_aligned_bounding_box()
    mn = aabb.get_min_bound()
    mx = aabb.get_max_bound()
    plane_pts = np.array([
        [mn[0], mn[1], Z_PLANE],
        [mx[0], mn[1], Z_PLANE],
        [mx[0], mx[1], Z_PLANE],
        [mn[0], mx[1], Z_PLANE],
    ], dtype=float)
    plane_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    plane = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(plane_pts),
        lines=o3d.utility.Vector2iVector(plane_lines),
    )
    vis.add_geometry(plane)

    # Start / Goal spheres
    s0 = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    s0.compute_vertex_normals()
    s0.translate(start_xz)
    s0.paint_uniform_color([0, 0, 1])
    vis.add_geometry(s0)

    sg = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    sg.compute_vertex_normals()
    sg.translate(goal_xz)
    sg.paint_uniform_color([0, 1, 0])
    vis.add_geometry(sg)

    # Path visualization (if any)
    if len(solution) > 0:
        path_pts = np.array([sol["pos"] for sol in solution], dtype=np.float64)
        path_pts[:, 2] = Z_PLANE

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(path_pts)
        line_set.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(path_pts) - 1)])
        vis.add_geometry(line_set)

        for p in path_pts:
            sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sp.compute_vertex_normals()
            sp.translate(p)
            sp.paint_uniform_color([0, 0, 0])
            vis.add_geometry(sp)
    else:
        path_pts = np.array([start_xz], dtype=np.float64)

    # --- robot boundary visualization ---
    def rotate_poly(poly_local, yaw):
        """Rotate 2D polygon vertices by yaw (radians) around origin."""
        c, s = np.cos(yaw), np.sin(yaw)
        R2 = np.array([[c, -s], [s, c]])
        return (R2 @ poly_local.T).T

    draw_slices = poly_slices
    if len(draw_slices) > MAX_POLY_SLICES_TO_DRAW:
        idxs = np.linspace(0, len(draw_slices) - 1, MAX_POLY_SLICES_TO_DRAW).astype(int)
        draw_slices = [draw_slices[i] for i in idxs]

    if SHOW_ROBOT_POLYS_AT_START_GOAL:
        start_yaw_val = quat_to_yaw(start["quat"])
        goal_yaw_val = quat_to_yaw(goal["quat"])
        for z_off, poly_local in draw_slices:
            z_abs = Z_PLANE + z_off
            vis.add_geometry(make_poly_lineset(rotate_poly(poly_local, start_yaw_val) + start_xz[:2], z_abs, color=(1.0, 0.0, 0.0)))
            vis.add_geometry(make_poly_lineset(rotate_poly(poly_local, goal_yaw_val) + goal_xz[:2],  z_abs, color=(0.0, 1.0, 0.0)))

    if SHOW_ROBOT_POLYS_ALONG_PATH:
        for i, sol in enumerate(solution):
            if i % ROBOT_POLY_SUBSAMPLE != 0:
                continue
            xy = sol["pos"][:2]
            yaw_val = quat_to_yaw(sol["quat"])
            for z_off, poly_local in draw_slices:
                z_abs = Z_PLANE + z_off
                vis.add_geometry(make_poly_lineset(rotate_poly(poly_local, yaw_val) + xy, z_abs, color=(0.2, 0.2, 1.0)))

    # --- collision debug points ---
    if SHOW_COLLISION_POINTS:
        if coll_start.shape[0] > 0:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(coll_start)
            pc.paint_uniform_color([1.0, 0.0, 1.0])  # magenta
            vis.add_geometry(pc)

        if coll_goal.shape[0] > 0:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(coll_goal)
            pc.paint_uniform_color([1.0, 1.0, 0.0])  # yellow
            vis.add_geometry(pc)

    vis.run()


if __name__ == "__main__":
    main()
