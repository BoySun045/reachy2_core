import numpy as np
from scipy.spatial import KDTree

# ompl pybind
from ompl import base as ob
from ompl import geometric as og

from functools import partial


class PathPlanner:
    def __init__(self):
        self.sp = ob.RealVectorStateSpace(3)
        # pass a dummy bound to initialize the state space
        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, -5)
        bounds.setHigh(0, 5)
        bounds.setLow(1, -5)
        bounds.setHigh(1, 5)
        bounds.setLow(2, -5)
        bounds.setHigh(2, 5)
        self.sp.setBounds(bounds)

        self.vx_grid = None
        self.invx_grid = None
        self.use_invx = False
        self.vx_grid_size = 0.1
        self.collision_radius = self.vx_grid_size
        self.inv_radius = 0.1

        # start and goal
        self.start_state = None
        self.goal_state = None

        self.ss = og.SimpleSetup(self.sp)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(
            partial(PathPlanner.isStateValid, self)))
        self.sp.setup()
        self.ss.getSpaceInformation().setStateValidityCheckingResolution(0.01)  # default is also 0.01

    def use_state(self, use_invx):
        # whether to use the invalid voxel grid to check the validity of the state
        # or use the valid voxel grid to check the validity of the state
        self.use_invx = use_invx

    def update_collision_radius(self, collision_radius, inv_radius):
        self.collision_radius = collision_radius
        self.inv_radius = inv_radius

    def isStateValid(self, state):

        state_3dpt = np.array([state[0], state[1], state[2]])

        if not self.use_invx:
            assert self.vx_grid is not None, "valid voxel sp is not initialized"
            # check if the state is in the input voxelgrid, by checking the nearest neighbor using KDTree
            dist, _ = self.kdt.query(state_3dpt)
            return dist < self.collision_radius

        if self.use_invx:
            assert self.invx_grid is not None, "invalid voxel sp is not initialized"
            dist_2, _ = self.kdt_invx.query(state_3dpt)
            return dist_2 > self.inv_radius

    def update_sp(self, bound, valid_vx, invalid_vx,
                  input_vx_size=0.1):

        # first, get the bound of input voxelgrid for constructing the state space
        low_x, high_x = bound["low_x"], bound["high_x"]
        low_y, high_y = bound["low_y"], bound["high_y"]
        low_z, high_z = bound["low_z"], bound["high_z"]

        epsilon = 0.0
        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, low_x - epsilon)
        bounds.setHigh(0, high_x + epsilon)
        bounds.setLow(1, low_y - epsilon)
        bounds.setHigh(1, high_y + epsilon)
        bounds.setLow(2, low_z - epsilon)
        bounds.setHigh(2, high_z + epsilon)

        # update ss
        self.sp.setBounds(bounds)

        self.vx_grid = valid_vx
        self.vx_grid_size = input_vx_size
        if not self.use_invx:
            self.kdt = KDTree(self.vx_grid)

        # update invx_grid
        self.invx_grid = invalid_vx
        if self.use_invx:
            self.kdt_invx = KDTree(self.invx_grid)

    def update_start_goal(self, start, goal):

        self.start_pos = start["pos"].copy()
        self.goal_pos = goal["pos"].copy()
        self.start_quat = start["quat"].copy()
        self.goal_quat = goal["quat"].copy()

        start_state = ob.State(self.sp)
        start_state()[0] = float(self.start_pos[0])
        start_state()[1] = float(self.start_pos[1])
        start_state()[2] = float(self.start_pos[2])

        goal_state = ob.State(self.sp)
        goal_state()[0] = float(self.goal_pos[0])
        goal_state()[1] = float(self.goal_pos[1])
        goal_state()[2] = float(self.goal_pos[2])

        self.ss.setStartAndGoalStates(start_state, goal_state, 0.1)  # TODO: check what is the last parameter

        return True

    def get_bounds(self):
        bounds = self.sp.getBounds()
        low_x = bounds.low[0]
        high_x = bounds.high[0]
        low_y = bounds.low[1]
        high_y = bounds.high[1]
        low_z = bounds.low[2]
        high_z = bounds.high[2]
        return {"low_x": low_x, "high_x": high_x, "low_y": low_y, "high_y": high_y, "low_z": low_z,
                "high_z": high_z}

    def get_start_goal(self):
        return {"start": {"pos": self.start_pos, "quat": self.start_quat},
                "goal": {"pos": self.goal_pos, "quat": self.goal_quat}}

    def get_motion_check_resolution(self):
        return self.ss.getSpaceInformation().getStateValidityCheckingResolution()

    def solve(self, time_limit=5.0, method="rrtstar"):

        if method == "rrtstar":
            self.ss.setPlanner(og.RRTstar(self.ss.getSpaceInformation()))
        elif method == "rrtconnect":
            self.ss.setPlanner(og.RRTConnect(self.ss.getSpaceInformation()))
        elif method == "rrt":
            self.ss.setPlanner(og.RRT(self.ss.getSpaceInformation()))

        solved = self.ss.solve(time_limit)
        if solved:
            # try to shorten the path
            self.ss.simplifySolution()
            pass
            # print(self.ss.getSolutionPath())
        else:
            print("Solver failed to find a solution")

    def get_solution(self):

        try:
            path = self.ss.getSolutionPath()
            path_states = path.getStates()
            solution = []
            for i, state in enumerate(path_states):

                pos = [state[0], state[1], state[2]]
                quat = [0, 0, 0, 1]

                if i == len(path_states) - 1:
                    # set the rotation same as the goal rotation
                    quat = [self.goal_quat[0], self.goal_quat[1], self.goal_quat[2], self.goal_quat[3]]
                solution.append({"pos": pos, "quat": quat})
            return solution

        except:
            print("No solution found")
            return None

    # ============================================================
    # NEW: switch OMPL validity checker to a NEW function
    # ============================================================
    def use_validity_checker(self, which="default"):
        """
        which:
          - "default": use existing isStateValid
          - "poly_stack": use polygon-stack checker (isStateValid_poly_stack)
        """
        if which == "default":
            fn = partial(PathPlanner.isStateValid, self)
        elif which == "poly_stack":
            fn = partial(PathPlanner.isStateValid_poly_stack, self)
        else:
            raise ValueError(f"Unknown validity checker: {which}")

        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(fn))

    # ============================================================
    # NEW: configure height-dependent polygon stack
    # ============================================================
    def set_robot_polygon_stack(
        self,
        slices,
        *,
        z_mode="offset",   # "offset": z_abs = state_z + z_value, "abs": z_abs = z_value
        z_band=0.25,       # consider obstacle voxels with |z - z_abs| <= z_band/2
        margin=0.05        # collision if obstacle point is inside poly OR within margin of an edge
    ):
        """
        slices: list of (z_value, poly_xy)
          poly_xy: iterable of (x,y) vertices in ROBOT LOCAL frame (centered at state XY).
        """
        if z_mode not in ("offset", "abs"):
            raise ValueError("z_mode must be 'offset' or 'abs'")

        self._poly_z_mode = z_mode
        self._poly_z_band = float(z_band)
        self._poly_margin = float(margin)

        self._poly_slices = []
        for z_value, poly_xy in slices:
            poly = np.asarray(poly_xy, dtype=np.float64)
            if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
                raise ValueError("Each polygon must be shape (M,2) with M >= 3")
            self._poly_slices.append((float(z_value), poly))

        # cache invx array & z for faster checks (if already initialized)
        if hasattr(self, "invx_grid") and self.invx_grid is not None:
            self._cache_invx_points()

    # ============================================================
    # NEW: cache invalid voxel points (speeds up repeated checks)
    # ============================================================
    def _cache_invx_points(self):
        assert self.invx_grid is not None, "invalid voxel sp is not initialized"
        self._invx_pts = np.asarray(self.invx_grid, dtype=np.float64)
        self._invx_z = self._invx_pts[:, 2]

    # ============================================================
    # NEW: vectorized point-in-polygon (ray casting)
    # ============================================================
    def _points_in_poly(self, points_xy: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
        """
        points_xy: (N,2), poly_xy: (M,2)
        returns: (N,) bool
        """
        x = points_xy[:, 0]
        y = points_xy[:, 1]
        xp = poly_xy[:, 0]
        yp = poly_xy[:, 1]
        m = poly_xy.shape[0]

        inside = np.zeros(points_xy.shape[0], dtype=bool)
        j = m - 1
        for i in range(m):
            xi, yi = xp[i], yp[i]
            xj, yj = xp[j], yp[j]
            denom = (yj - yi)
            denom = denom if abs(denom) > 1e-12 else (1e-12 if denom >= 0 else -1e-12)
            hit = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / denom + xi)
            inside ^= hit
            j = i
        return inside

    # ============================================================
    # NEW: distance from points to polygon edges (for margin)
    # ============================================================
    def _point_segment_dist_sq(self, P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        P: (N,2), A/B: (2,)
        returns: (N,) squared distances
        """
        AB = B - A
        AP = P - A
        denom = float(np.dot(AB, AB)) + 1e-12
        t = (AP @ AB) / denom
        t = np.clip(t, 0.0, 1.0)
        proj = A + np.outer(t, AB)
        d = P - proj
        return np.sum(d * d, axis=1)

    def _poly_collides_points(self, points_xy: np.ndarray, poly_xy: np.ndarray, margin: float) -> bool:
        """
        collision if any point is inside polygon OR within margin of any edge
        """
        if points_xy.shape[0] == 0:
            return False

        if np.any(self._points_in_poly(points_xy, poly_xy)):
            return True

        if margin <= 0:
            return False

        m2 = margin * margin
        M = poly_xy.shape[0]
        for i in range(M):
            A = poly_xy[i]
            B = poly_xy[(i + 1) % M]
            d2 = self._point_segment_dist_sq(points_xy, A, B)
            if np.any(d2 <= m2):
                return True
        return False

    # ============================================================
    # NEW: polygon-stack validity checker (height-dependent)
    # ============================================================
    def isStateValid_poly_stack(self, state):
        """
        Uses invalid voxels (invx_grid). Requires:
          - self.use_invx == True
          - set_robot_polygon_stack(...) called
        """
        if not self.use_invx:
            raise RuntimeError("Polygon stack checker expects use_invx=True (invalid voxels).")
        assert self.invx_grid is not None, "invalid voxel sp is not initialized"
        assert hasattr(self, "_poly_slices"), "Call set_robot_polygon_stack(...) first."

        # cache invx points if not already
        if not hasattr(self, "_invx_pts"):
            self._cache_invx_points()

        x = float(state[0])
        y = float(state[1])
        z_state = float(state[2])

        half = 0.5 * self._poly_z_band
        margin = self._poly_margin

        # Treat obstacle voxel centers as points and test them against the polygon footprint per slice.
        for z_value, poly_local in self._poly_slices:
            z_abs = (z_state + z_value) if (self._poly_z_mode == "offset") else z_value

            # select obstacles in z band around this slice
            zmask = (self._invx_z >= (z_abs - half)) & (self._invx_z <= (z_abs + half))
            cand = self._invx_pts[zmask]
            if cand.shape[0] == 0:
                continue

            # move polygon from robot-local to world XY
            poly_world = poly_local + np.array([x, y], dtype=np.float64)

            # AABB prefilter in XY for speed
            mn = np.min(poly_world, axis=0) - margin
            mx = np.max(poly_world, axis=0) + margin
            pts_xy = cand[:, :2]
            aabb = (pts_xy[:, 0] >= mn[0]) & (pts_xy[:, 0] <= mx[0]) & \
                   (pts_xy[:, 1] >= mn[1]) & (pts_xy[:, 1] <= mx[1])
            pts_xy = pts_xy[aabb]
            if pts_xy.shape[0] == 0:
                continue

            # collision if any obstacle point lies inside/near the polygon
            if self._poly_collides_points(pts_xy, poly_world, margin):
                return False

        return True

    # ============================================================
    # NEW (Option B): XYZYAW planning + yaw-aware polygon collision
    # ============================================================

    @staticmethod
    def _yaw_wrap(yaw: float) -> float:
        return (yaw + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _quat_from_yaw(yaw: float) -> np.ndarray:
        return np.array([0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)], dtype=np.float64)

    @staticmethod
    def _rotate_xy(points_xy: np.ndarray, yaw: float) -> np.ndarray:
        c = np.cos(yaw)
        s = np.sin(yaw)
        R = np.array([[c, -s],
                      [s,  c]], dtype=np.float64)
        return points_xy @ R.T

    def init_space_xyzyaw(self, yaw_low=-np.pi, yaw_high=np.pi):
        """
        Re-initialize OMPL state space to 4D: [x, y, z, yaw].
        Does not modify any existing 3D method; you opt-in by calling this.
        """
        self.sp = ob.RealVectorStateSpace(4)

        bounds = ob.RealVectorBounds(4)
        # dummy init bounds (will be overwritten)
        bounds.setLow(0, -5); bounds.setHigh(0, 5)   # x
        bounds.setLow(1, -5); bounds.setHigh(1, 5)   # y
        bounds.setLow(2, -5); bounds.setHigh(2, 5)   # z
        bounds.setLow(3, float(yaw_low)); bounds.setHigh(3, float(yaw_high))  # yaw
        self.sp.setBounds(bounds)

        self.ss = og.SimpleSetup(self.sp)
        # default validity checker = existing isStateValid (sphere invx), unless you change it
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(
            partial(PathPlanner.isStateValid, self)
        ))

        self.sp.setup()
        self.ss.getSpaceInformation().setStateValidityCheckingResolution(0.01)

    def update_sp_xyzyaw(self, bound, valid_vx, invalid_vx, input_vx_size=0.1,
                         yaw_low=-np.pi, yaw_high=np.pi):
        """
        Like update_sp(), but for 4D space: [x,y,z,yaw].
        """
        low_x, high_x = bound["low_x"], bound["high_x"]
        low_y, high_y = bound["low_y"], bound["high_y"]
        low_z, high_z = bound["low_z"], bound["high_z"]

        bounds = ob.RealVectorBounds(4)
        bounds.setLow(0, float(low_x));  bounds.setHigh(0, float(high_x))
        bounds.setLow(1, float(low_y));  bounds.setHigh(1, float(high_y))
        bounds.setLow(2, float(low_z));  bounds.setHigh(2, float(high_z))
        bounds.setLow(3, float(yaw_low)); bounds.setHigh(3, float(yaw_high))
        self.sp.setBounds(bounds)

        self.vx_grid = valid_vx
        self.vx_grid_size = input_vx_size
        if not self.use_invx:
            self.kdt = KDTree(self.vx_grid)

        self.invx_grid = invalid_vx
        if self.use_invx:
            self.kdt_invx = KDTree(self.invx_grid)

        # If poly stack already configured, refresh invx cache
        if hasattr(self, "_poly_slices"):
            self._cache_invx_points()

    def update_start_goal_xyzyaw(self, start, goal):
        """
        start/goal dicts:
          start["pos"] = [x,y,z]
          start["yaw"] = yaw radians (optional, default 0)
        """
        self.start_pos = start["pos"].copy()
        self.goal_pos = goal["pos"].copy()
        self.start_yaw = float(start.get("yaw", 0.0))
        self.goal_yaw = float(goal.get("yaw", 0.0))

        self.start_quat = self._quat_from_yaw(self.start_yaw)
        self.goal_quat = self._quat_from_yaw(self.goal_yaw)

        start_state = ob.State(self.sp)
        start_state()[0] = float(self.start_pos[0])
        start_state()[1] = float(self.start_pos[1])
        start_state()[2] = float(self.start_pos[2])
        start_state()[3] = float(self._yaw_wrap(self.start_yaw))

        goal_state = ob.State(self.sp)
        goal_state()[0] = float(self.goal_pos[0])
        goal_state()[1] = float(self.goal_pos[1])
        goal_state()[2] = float(self.goal_pos[2])
        goal_state()[3] = float(self._yaw_wrap(self.goal_yaw))

        self.ss.setStartAndGoalStates(start_state, goal_state, 0.1)
        return True

    def use_validity_checker_xyzyaw(self, which="invx_sphere"):
        """
        which:
          - "invx_sphere": use existing isStateValid (ignores yaw)
          - "poly_stack_yaw": rotate polygon footprint by yaw
        """
        if which == "invx_sphere":
            fn = partial(PathPlanner.isStateValid, self)
        elif which == "poly_stack_yaw":
            fn = partial(PathPlanner.isStateValid_poly_stack_yaw, self)
        else:
            raise ValueError(f"Unknown validity checker: {which}")

        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(fn))

    def isStateValid_poly_stack_yaw(self, state):
        """
        Yaw-aware polygon-stack validity.
        state = [x,y,z,yaw] (OMPL state supports indexing).
        Requires:
          - use_invx=True
          - set_robot_polygon_stack(...) called
        """
        if not self.use_invx:
            raise RuntimeError("Yaw polygon stack checker expects use_invx=True (invalid voxels).")
        assert self.invx_grid is not None, "invalid voxel sp is not initialized"
        assert hasattr(self, "_poly_slices"), "Call set_robot_polygon_stack(...) first."

        if not hasattr(self, "_invx_pts"):
            self._cache_invx_points()

        x = float(state[0])
        y = float(state[1])
        z_state = float(state[2])
        yaw = float(state[3])

        half = 0.5 * self._poly_z_band
        margin = self._poly_margin

        for z_value, poly_local in self._poly_slices:
            z_abs = (z_state + z_value) if (self._poly_z_mode == "offset") else z_value

            zmask = (self._invx_z >= (z_abs - half)) & (self._invx_z <= (z_abs + half))
            cand = self._invx_pts[zmask]
            if cand.shape[0] == 0:
                continue

            # Rotate local footprint by yaw, then translate to world xy
            poly_rot = self._rotate_xy(poly_local, yaw)
            poly_world = poly_rot + np.array([x, y], dtype=np.float64)

            # AABB filter in XY
            mn = np.min(poly_world, axis=0) - margin
            mx = np.max(poly_world, axis=0) + margin
            pts_xy = cand[:, :2]
            aabb = (pts_xy[:, 0] >= mn[0]) & (pts_xy[:, 0] <= mx[0]) & \
                   (pts_xy[:, 1] >= mn[1]) & (pts_xy[:, 1] <= mx[1])
            pts_xy = pts_xy[aabb]
            if pts_xy.shape[0] == 0:
                continue

            if self._poly_collides_points(pts_xy, poly_world, margin):
                return False

        return True

    def get_solution_xyzyaw(self):
        """
        Returns:
          [{"pos":[x,y,z], "yaw":yaw, "quat":[qx,qy,qz,qw]}, ...]
        """
        try:
            path = self.ss.getSolutionPath()
            path_states = path.getStates()
            solution = []

            for i, state in enumerate(path_states):
                x = float(state[0])
                y = float(state[1])
                z = float(state[2])
                yaw = float(state[3])
                yaw = self._yaw_wrap(yaw)
                quat = self._quat_from_yaw(yaw)

                # last state orientation = goal yaw (if provided)
                if i == len(path_states) - 1 and hasattr(self, "goal_yaw"):
                    yaw = self._yaw_wrap(float(self.goal_yaw))
                    quat = self._quat_from_yaw(yaw)

                solution.append({
                    "pos": [x, y, z],
                    "yaw": float(yaw),
                    "quat": quat.tolist(),
                })

            return solution

        except Exception:
            print("No solution found")
            return None
