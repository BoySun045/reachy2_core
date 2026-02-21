#!/usr/bin/env python3
"""
ROS2 node that publishes a TSDF scene point cloud and allows
terminal-based object queries with pose + bounding box visualization.

Usage:
    python3 pathplanner_manager.py
"""

import json
import os
import sys
import argparse
import gzip
import pickle
import random
import re
import threading

import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, String

import open3d as o3d

FRAME_ID = "map"
MAX_ARM_REACH = 0.80  # metres from base centre


def load_config(config_path: str) -> dict:
    """Load YAML config and resolve paths relative to the config file directory."""
    config_path = os.path.abspath(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    config_dir = os.path.dirname(config_path)
    data_dir = os.path.join(config_dir, cfg["data_dir"])
    cfg["_data_dir"] = data_dir
    cfg["_pcd_path"] = os.path.join(data_dir, cfg["pcd_file"])
    cfg["_reachability_path"] = os.path.join(data_dir, cfg["reachability_file"])
    cfg["_objects_path"] = os.path.join(data_dir, cfg["objects_file"])
    return cfg


class PathPlannerManagerNode(Node):
    def __init__(self, cfg: dict):
        super().__init__("pathplanner_manager")

        self._planning_mode = cfg.get("planning_mode", "collision")
        self.get_logger().info(f"Planning mode: {self._planning_mode}")

        pcd_path = cfg["_pcd_path"]
        reachability_path = cfg["_reachability_path"]
        objects_path = cfg["_objects_path"]

        # --- Load TSDF point cloud ---
        self.get_logger().info(f"Loading point cloud from {pcd_path}")
        pcd = o3d.io.read_point_cloud(pcd_path)
        if len(pcd.points) == 0:
            raise RuntimeError(f"Empty point cloud: {pcd_path}")
        self.get_logger().info(f"Loaded {len(pcd.points)} points")
        self._pcd_msg = self._build_pointcloud2_msg(pcd)

        # --- Load reachability map ---
        self.get_logger().info(f"Loading reachability from {reachability_path}")
        pcd_reach = o3d.io.read_point_cloud(reachability_path)
        self._reachable_pts = np.asarray(pcd_reach.points)
        self.get_logger().info(
            f"Loaded {len(self._reachable_pts)} reachable points"
        )

        # --- Load objects ---
        self.get_logger().info(f"Loading objects from {objects_path}")
        with gzip.open(objects_path, "rb") as f:
            data = pickle.load(f)
        self._objects = data["objects"]
        if cfg.get("z_up", False):
            # Convert bboxes from Y-up to Z-up: (x, y, z) -> (x, z, -y)
            for obj in self._objects:
                bbox = np.array(obj["bbox_np"], dtype=np.float64)
                y = bbox[:, 1].copy()
                bbox[:, 1] = bbox[:, 2]
                bbox[:, 2] = -y
                obj["bbox_np"] = bbox
            self.get_logger().info("Object bboxes converted to Z-up")
        self.get_logger().info(f"Loaded {len(self._objects)} objects")

        # --- Publishers ---
        self._pcd_pub = self.create_publisher(
            PointCloud2, "~/scene_pointcloud", 10
        )

        latched_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self._pose_pub = self.create_publisher(
            PoseStamped, "~/object_pose", latched_qos
        )
        self._bbox_pub = self.create_publisher(
            Marker, "~/object_bbox", latched_qos
        )
        self._reach_pub = self.create_publisher(
            PointCloud2, "~/reachability", 10
        )

        # Per-robot nav goal publishers
        self._goal_pubs = {
            "reachy": self.create_publisher(
                PoseStamped, "~/nav_goal", latched_qos
            ),
            "spot": self.create_publisher(
                PoseStamped, "~/spot/nav_goal", latched_qos
            ),
        }

        # --- Build reachability PointCloud2 (green) ---
        self._reach_msg = self._build_pointcloud2_msg(pcd_reach)

        # --- Timer for point cloud publishing at 0.1 Hz ---
        self.create_timer(10.0, self._publish_pointcloud)

        # --- Subscriber for LLM-routed queries ---
        self.create_subscription(
            String, '~/query', self._on_query, 10
        )

        # --- Print available objects ---
        self._print_object_list()

        # --- Start terminal query thread ---
        self._query_thread = threading.Thread(
            target=self._query_loop, daemon=True
        )
        self._query_thread.start()

        self.get_logger().info("PathPlannerManager ready.")

    # ------------------------------------------------------------------
    # Point cloud
    # ------------------------------------------------------------------
    def _build_pointcloud2_msg(self, pcd):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        pts = np.asarray(pcd.points, dtype=np.float32)  # (N, 3)
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)  # (N, 3)
            rgb_uint32 = (
                colors[:, 0].astype(np.uint32) << 16
                | colors[:, 1].astype(np.uint32) << 8
                | colors[:, 2].astype(np.uint32)
            )
            rgb_float32 = rgb_uint32.view(np.float32)
        else:
            rgb_float32 = np.full(pts.shape[0], 0.5, dtype=np.float32)

        xyzrgb = np.column_stack([pts, rgb_float32.reshape(-1, 1)])

        header = self._make_header()
        return pc2.create_cloud(header, fields, xyzrgb)

    def _publish_pointcloud(self):
        now = self.get_clock().now().to_msg()
        self._pcd_msg.header.stamp = now
        self._pcd_pub.publish(self._pcd_msg)
        self._reach_msg.header.stamp = now
        self._reach_pub.publish(self._reach_msg)

    # ------------------------------------------------------------------
    # Object pose from bounding box
    # ------------------------------------------------------------------
    def _compute_object_pose(self, obj):
        bbox = obj["bbox_np"]
        centroid = bbox.mean(axis=0)

        # PCA to get oriented bounding box axes
        vecs = bbox - centroid
        _, _, Vt = np.linalg.svd(vecs, full_matrices=False)
        R_mat = Vt.T
        if np.linalg.det(R_mat) < 0:
            R_mat[:, 2] = -R_mat[:, 2]
        quat = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]

        ps = PoseStamped()
        ps.header = self._make_header()
        ps.pose.position.x = float(centroid[0])
        ps.pose.position.y = float(centroid[1])
        ps.pose.position.z = float(centroid[2])
        ps.pose.orientation.x = float(quat[0])
        ps.pose.orientation.y = float(quat[1])
        ps.pose.orientation.z = float(quat[2])
        ps.pose.orientation.w = float(quat[3])
        return ps

    # ------------------------------------------------------------------
    # Navigation goal from reachability
    # ------------------------------------------------------------------
    def _compute_nav_goal(self, object_pos):
        """Find the closest reachable point to the object, oriented with
        robot X facing the object.  In collision mode, only considers points
        within MAX_ARM_REACH.  Returns PoseStamped or None."""
        obj_xy = object_pos[:2]
        dists = np.linalg.norm(self._reachable_pts[:, :2] - obj_xy, axis=1)

        if self._planning_mode == "collision":
            within = dists <= MAX_ARM_REACH
            if not np.any(within):
                return None
            best = np.argmin(dists[within])
            goal_pt = self._reachable_pts[within][best]
        else:
            best = np.argmin(dists)
            goal_pt = self._reachable_pts[best]

        dx = obj_xy[0] - goal_pt[0]
        dy = obj_xy[1] - goal_pt[1]
        yaw = np.arctan2(dy, dx)
        quat = R.from_euler("z", yaw).as_quat()  # [x, y, z, w]

        ps = PoseStamped()
        ps.header = self._make_header()
        ps.pose.position.x = float(goal_pt[0])
        ps.pose.position.y = float(goal_pt[1])
        ps.pose.position.z = float(goal_pt[2])
        ps.pose.orientation.x = float(quat[0])
        ps.pose.orientation.y = float(quat[1])
        ps.pose.orientation.z = float(quat[2])
        ps.pose.orientation.w = float(quat[3])
        return ps

    # ------------------------------------------------------------------
    # Bounding box marker
    # ------------------------------------------------------------------
    def _build_bbox_marker(self, obj):
        bbox = obj["bbox_np"]
        centroid = bbox.mean(axis=0)

        # Find 12 edges via PCA sign pattern
        vecs = bbox - centroid
        _, _, Vt = np.linalg.svd(vecs, full_matrices=False)
        projs = vecs @ Vt.T
        signs = np.sign(projs)

        edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(signs[i] != signs[j]) == 1:
                    edges.append((i, j))

        marker = Marker()
        marker.header = self._make_header()
        marker.ns = "object_bbox"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # line width
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        marker.pose.orientation.w = 1.0

        for i, j in edges:
            p1 = Point(
                x=float(bbox[i, 0]),
                y=float(bbox[i, 1]),
                z=float(bbox[i, 2]),
            )
            p2 = Point(
                x=float(bbox[j, 0]),
                y=float(bbox[j, 1]),
                z=float(bbox[j, 2]),
            )
            marker.points.append(p1)
            marker.points.append(p2)

        return marker

    # ------------------------------------------------------------------
    # Object query (shared by terminal and ROS subscriber)
    # ------------------------------------------------------------------
    def _find_and_publish_object(self, query: str,
                                robot: str = "reachy") -> bool:
        """Find an object by partial name match and publish pose + nav goal.

        If query ends with a number (e.g. "robot dog 2"), selects that
        specific instance (1-indexed).  Otherwise picks randomly.
        The nav goal is published to the robot-specific topic.
        Returns True if an object was found and published.
        """
        # Check for trailing index: "robot dog 2" → name="robot dog", idx=2
        m = re.match(r'^(.+?)\s+(\d+)$', query.strip())
        if m:
            name_part = m.group(1)
            requested_idx = int(m.group(2))
        else:
            name_part = query.strip()
            requested_idx = None

        matches = [
            obj
            for obj in self._objects
            if name_part.lower() in obj["name"].lower()
        ]

        if not matches:
            self.get_logger().warn(f"No objects matching '{name_part}'")
            return False

        if requested_idx is not None:
            if requested_idx < 1 or requested_idx > len(matches):
                self.get_logger().warn(
                    f"Index {requested_idx} out of range "
                    f"(have {len(matches)} '{name_part}' objects)"
                )
                return False
            selected = matches[requested_idx - 1]
            self.get_logger().info(
                f"Selected '{selected['name']}' #{requested_idx} "
                f"of {len(matches)}"
            )
        elif len(matches) > 1:
            selected = random.choice(matches)
            self.get_logger().info(
                f"Found {len(matches)} '{name_part}' objects, "
                f"randomly selected '{selected['name']}'"
            )
        else:
            selected = matches[0]

        self._publish_object(selected, robot=robot)
        return True

    def _on_query(self, msg):
        """Handle object query from LLM command router.

        Message is JSON: {"robot": "reachy"|"spot", "object": "..."}
        """
        raw = msg.data.strip()
        if not raw:
            return
        try:
            data = json.loads(raw)
            robot = data.get("robot", "reachy")
            query = data.get("object", "").strip()
        except (json.JSONDecodeError, AttributeError):
            # Fallback: treat as plain-text query (terminal / manual publish)
            robot = "reachy"
            query = raw
        if not query:
            return
        self.get_logger().info(
            f"Received query from LLM: robot={robot}, object='{query}'"
        )
        self._find_and_publish_object(query, robot=robot)

    # ------------------------------------------------------------------
    # Terminal query loop
    # ------------------------------------------------------------------
    def _query_loop(self):
        while rclpy.ok():
            try:
                query = input(
                    "\nEnter object name (or 'list' to show all): "
                ).strip()
            except EOFError:
                break

            if not query:
                continue

            if query.lower() == "list":
                self._print_object_list()
                continue

            # Case-insensitive partial match
            matches = [
                obj
                for obj in self._objects
                if query.lower() in obj["name"].lower()
            ]

            if len(matches) == 0:
                print(f"  No objects matching '{query}'")
                continue

            if len(matches) == 1:
                selected = matches[0]
            else:
                print(f"  Found {len(matches)} matches:")
                for idx, obj in enumerate(matches):
                    c = obj["bbox_np"].mean(axis=0)
                    print(
                        f"    [{idx}] {obj['name']} "
                        f"(detections={obj['num_detections']}, "
                        f"pos=[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}])"
                    )
                try:
                    choice = input(
                        "  Pick index (or Enter for first): "
                    ).strip()
                    if choice == "":
                        selected = matches[0]
                    else:
                        selected = matches[int(choice)]
                except (ValueError, IndexError, EOFError):
                    print("  Invalid selection.")
                    continue

            self._publish_object(selected, robot="reachy")

    def _publish_object(self, obj, robot: str = "reachy"):
        centroid = obj["bbox_np"].mean(axis=0)
        self.get_logger().info(
            f"[{robot}] Publishing '{obj['name']}' at "
            f"[{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]"
        )
        self._pose_pub.publish(self._compute_object_pose(obj))
        self._bbox_pub.publish(self._build_bbox_marker(obj))
        print(f"  -> Published pose on ~/object_pose")
        print(f"  -> Published bbox on ~/object_bbox")

        goal = self._compute_nav_goal(centroid)
        if goal is not None:
            goal_pub = self._goal_pubs.get(robot)
            if goal_pub is None:
                self.get_logger().warn(
                    f"No nav_goal publisher for robot '{robot}'"
                )
                return
            goal_pub.publish(goal)
            p = goal.pose.position
            yaw = R.from_quat([
                goal.pose.orientation.x, goal.pose.orientation.y,
                goal.pose.orientation.z, goal.pose.orientation.w,
            ]).as_euler("xyz")[2]
            topic_name = goal_pub.topic_name
            print(
                f"  -> Nav goal: [{p.x:.3f}, {p.y:.3f}, {p.z:.3f}] "
                f"yaw={np.degrees(yaw):.1f}° on {topic_name}"
            )
        else:
            print(
                f"  ** No reachable point within {MAX_ARM_REACH}m of object"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_header(self):
        from std_msgs.msg import Header

        h = Header()
        h.frame_id = FRAME_ID
        h.stamp = self.get_clock().now().to_msg()
        return h

    def _print_object_list(self):
        print("\n--- Available objects ---")
        for idx, obj in enumerate(self._objects):
            c = obj["bbox_np"].mean(axis=0)
            print(
                f"  [{idx:3d}] {obj['name']:25s} "
                f"detections={obj['num_detections']:3d}  "
                f"pos=[{c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}]"
            )
        print(f"--- Total: {len(self._objects)} objects ---\n")


def main():
    parser = argparse.ArgumentParser(description="PathPlanner Manager Node")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)
    cfg = load_config(args.config)
    node = PathPlannerManagerNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
