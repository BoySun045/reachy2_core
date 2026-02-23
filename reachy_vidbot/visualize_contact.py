#!/usr/bin/env python3
"""Visualize VidBot contact and goal heatmaps overlaid on the RGB image."""
import argparse
import glob
import os

import cv2
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")


def overlay_heatmap(rgb, heatmap, alpha=0.5):
    """Overlay a normalized heatmap on an RGB image."""
    hm = heatmap.copy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(rgb, 1 - alpha, hm_color, alpha, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="spot")
    parser.add_argument("-f", "--frame", default="000000")
    args = parser.parse_args()

    ds = os.path.join(DATASET_DIR, args.dataset)
    color_path = os.path.join(ds, "color", f"{args.frame}.png")
    pred_dir = os.path.join(ds, "prediction")

    npzs = sorted(glob.glob(os.path.join(pred_dir, "*.npz")), key=os.path.getmtime)
    if not npzs:
        print("No prediction NPZ found")
        return
    npz_path = npzs[-1]
    print(f"Loading: {npz_path}")

    z = np.load(npz_path, allow_pickle=True)
    rgb = cv2.imread(color_path)
    h, w = rgb.shape[:2]

    # --- Contact heatmap (256x256 crop → map back to full image via bbox) ---
    contact_scores = z["contact_scores"][0]        # (256, 256)
    bbox = z["bbox"][0].astype(int)                # x1, y1, x2, y2 in raw image coords
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    contact_overlay = rgb.copy()
    if bw > 0 and bh > 0:
        hm_resized = cv2.resize(contact_scores, (bw, bh))
        hm_norm = (hm_resized - hm_resized.min()) / (hm_resized.max() - hm_resized.min() + 1e-8)
        hm_color = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # Clip to image bounds
        px1, py1 = max(0, x1), max(0, y1)
        px2, py2 = min(w, x2), min(h, y2)
        hx1, hy1 = px1 - x1, py1 - y1
        hx2, hy2 = hx1 + (px2 - px1), hy1 + (py2 - py1)
        contact_overlay[py1:py2, px1:px2] = cv2.addWeighted(
            contact_overlay[py1:py2, px1:px2], 0.5,
            hm_color[hy1:hy2, hx1:hx2], 0.5, 0,
        )
        cv2.rectangle(contact_overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Draw contact point + samples
    contact_pix = z["contact_pix"][0, 0]
    cx, cy = int(contact_pix[0]), int(contact_pix[1])
    for s in z["contact_pix_samples"][0]:
        cv2.circle(contact_overlay, (int(s[0]), int(s[1])), 2, (0, 200, 0), -1)
    cv2.circle(contact_overlay, (cx, cy), 8, (0, 255, 0), 2)
    cv2.putText(contact_overlay, "contact", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Goal heatmap (256x448 resized → scale back to full image) ---
    goal_heatmap = z["goal_heatmap"][0]            # (256, 448)
    goal_hm_full = cv2.resize(goal_heatmap, (w, h))
    goal_overlay = overlay_heatmap(rgb, goal_hm_full, alpha=0.5)

    # Draw goal point + samples
    goal_pix = z["goal_pix"][0]
    gx, gy = int(goal_pix[0]), int(goal_pix[1])
    for s in z["goal_pix_samples"][0]:
        cv2.circle(goal_overlay, (int(s[0]), int(s[1])), 2, (200, 0, 0), -1)
    cv2.circle(goal_overlay, (gx, gy), 8, (255, 0, 0), 2)
    cv2.putText(goal_overlay, "goal", (gx + 10, gy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Also draw contact on goal view for reference
    cv2.circle(goal_overlay, (cx, cy), 6, (0, 255, 0), 1)

    # --- Side by side ---
    combined = np.hstack([contact_overlay, goal_overlay])
    cv2.putText(combined, "Contact", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(combined, "Goal", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    out_path = os.path.join(ds, "contact_heatmap.png")
    cv2.imwrite(out_path, combined)
    print(f"Saved: {out_path}")

    cv2.imshow("VidBot Heatmaps (Contact | Goal)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
