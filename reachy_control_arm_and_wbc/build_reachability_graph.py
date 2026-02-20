#!/usr/bin/env python3
"""
Build a reachability graph from the sampled workspace.

For each feasible EE pose in the CSV, finds all other feasible poses
within a configurable radius and saves the adjacency list as a
compressed numpy file (.npz).

Usage:
  python3 build_reachability_graph.py [--radius 0.05] [--csv ee_poses_right.csv]

Output:
  reachability_graph_right.npz  containing:
    - positions    (N, 3)       EE positions
    - quaternions  (N, 4)       EE orientations
    - joints       (N, 7)       Joint configurations
    - adj_indices  (M,)         Flattened neighbour indices
    - adj_offsets  (N+1,)       CSR-style offsets into adj_indices
                                 neighbours of point i = adj_indices[adj_offsets[i]:adj_offsets[i+1]]
    - adj_dists    (M,)         Corresponding distances
    - radius       scalar       The radius used
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

R_ARM_JOINTS = [
    "r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw", "r_elbow_pitch",
    "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw",
]
L_ARM_JOINTS = [
    "l_shoulder_pitch", "l_shoulder_roll", "l_elbow_yaw", "l_elbow_pitch",
    "l_wrist_roll", "l_wrist_pitch", "l_wrist_yaw",
]


def load_csv(path: str, arm: str):
    joints_key = R_ARM_JOINTS if arm == "right" else L_ARM_JOINTS
    joint_configs, positions, quats = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            joint_configs.append([float(row[j]) for j in joints_key])
            positions.append([float(row["ee_x"]), float(row["ee_y"]), float(row["ee_z"])])
            quats.append([float(row[k]) for k in ("ee_qx", "ee_qy", "ee_qz", "ee_qw")])
    return np.array(joint_configs), np.array(positions), np.array(quats)


def build_graph(positions: np.ndarray, radius: float):
    """Return CSR-style adjacency: (indices, offsets, distances)."""
    tree = KDTree(positions)
    neighbours = tree.query_ball_tree(tree, r=radius)

    all_indices = []
    all_dists = []
    offsets = [0]

    for i, nbrs in enumerate(neighbours):
        # Remove self
        nbrs = [j for j in nbrs if j != i]
        if nbrs:
            dists = np.linalg.norm(positions[nbrs] - positions[i], axis=1)
            # Sort by distance
            order = np.argsort(dists)
            nbrs = [nbrs[k] for k in order]
            dists = dists[order]
            all_indices.extend(nbrs)
            all_dists.extend(dists)
        offsets.append(len(all_indices))

    return (
        np.array(all_indices, dtype=np.int32),
        np.array(offsets, dtype=np.int32),
        np.array(all_dists, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description="Build workspace reachability graph")
    parser.add_argument("--csv", default="ee_poses_right.csv")
    parser.add_argument("--arm", default="right", choices=["right", "left"])
    parser.add_argument("--radius", type=float, default=0.05,
                        help="Neighbourhood radius in metres")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = csv_path.parent / f"reachability_graph_{args.arm}.npz"

    print(f"Loading {csv_path} ...")
    joints, positions, quats = load_csv(str(csv_path), args.arm)
    n = len(positions)
    print(f"  {n} samples loaded")

    print(f"Building reachability graph (radius={args.radius}m) ...")
    t0 = time.monotonic()
    adj_indices, adj_offsets, adj_dists = build_graph(positions, args.radius)
    elapsed = time.monotonic() - t0

    # Stats
    counts = np.diff(adj_offsets)
    total_edges = len(adj_indices)
    print(f"  Done in {elapsed:.2f}s")
    print(f"  Total edges:  {total_edges}  (avg {counts.mean():.1f} neighbours/point)")
    print(f"  Neighbour count:  min={counts.min()}  median={np.median(counts):.0f}"
          f"  max={counts.max()}  isolated={np.sum(counts == 0)}")

    np.savez_compressed(
        out_path,
        positions=positions,
        quaternions=quats,
        joints=joints,
        adj_indices=adj_indices,
        adj_offsets=adj_offsets,
        adj_dists=adj_dists,
        radius=np.float32(args.radius),
    )
    print(f"\nSaved to {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    print(f"\nTo query neighbours of point i:")
    print(f"  data = np.load('{out_path.name}')")
    print(f"  nbrs = data['adj_indices'][data['adj_offsets'][i]:data['adj_offsets'][i+1]]")


if __name__ == "__main__":
    main()
