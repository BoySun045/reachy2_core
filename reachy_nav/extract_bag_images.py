#!/usr/bin/env python3
"""
Extract all images from a ROS2 bag (CompressedImage topics) and save as PNG.

Usage:
    python3 extract_bag_images.py /path/to/rosbag2_folder

Creates one subdirectory per topic under an output folder next to the bag:
    <bag_name>_images/
    ├── camera_color/
    ├── camera_depth/
    ├── teleop_right/
    └── teleop_left/
"""

import sys
import os
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


# Topics to extract and their short names for output folders
IMAGE_TOPICS = {
    "/camera/color/image_raw/compressed": "camera_color",
    "/camera/depth/image_raw/compressed": "camera_depth",
    "/teleop_camera/right_image/image_raw/compressed": "teleop_right",
    "/teleop_camera/left_image/image_raw/compressed": "teleop_left",
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <rosbag_path> [output_dir]")
        sys.exit(1)

    bag_path = Path(sys.argv[1])
    if not bag_path.exists():
        print(f"Bag not found: {bag_path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        out_root = Path(sys.argv[2])
    else:
        out_root = bag_path.parent / f"{bag_path.name}_images"

    print(f"Bag:    {bag_path}")
    print(f"Output: {out_root}")

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with Reader(bag_path) as reader:
        # Find which of our target topics exist in this bag
        bag_topics = {c.topic for c in reader.connections}
        active_topics = {t: name for t, name in IMAGE_TOPICS.items() if t in bag_topics}

        if not active_topics:
            print("No matching image topics found in bag.")
            print(f"Available topics: {sorted(bag_topics)}")
            sys.exit(1)

        # Create output dirs
        for topic, folder_name in active_topics.items():
            (out_root / folder_name).mkdir(parents=True, exist_ok=True)

        print(f"Extracting from {len(active_topics)} topics:")
        for t, name in active_topics.items():
            print(f"  {t} -> {name}/")

        counts = {t: 0 for t in active_topics}

        connections = [c for c in reader.connections if c.topic in active_topics]

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            topic = connection.topic
            folder_name = active_topics[topic]

            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            if len(msg.data) == 0:
                continue
            img_data = np.frombuffer(msg.data, dtype=np.uint8)

            # Decode compressed image
            img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            idx = counts[topic]
            filename = out_root / folder_name / f"{idx:06d}.png"
            cv2.imwrite(str(filename), img)
            counts[topic] += 1

            total = sum(counts.values())
            if total % 500 == 0:
                print(f"  ... {total} images extracted so far")

    print(f"\nDone! Extracted:")
    for topic, n in counts.items():
        folder_name = active_topics[topic]
        print(f"  {folder_name}: {n} images")
    print(f"Total: {sum(counts.values())} images in {out_root}")


if __name__ == "__main__":
    main()
