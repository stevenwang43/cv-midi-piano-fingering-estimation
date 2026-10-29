"""
Processes raw hand-tracking and MIDI data files by applying camera lens 
undistortion, normalization, hand-scale correction, and calibration mapping, 
then compiles them into clean.csv.
"""

import glob
import json
import os
import cv2
import numpy as np
import pandas as pd


def load_calibration():
    if os.path.exists("calibration_config.json"):
        with open("calibration_config.json", "r") as f:
            return json.load(f)
    print("Warning: calibration_config.json not found. Using default calibration.")
    return {"norm_slope": 0.015, "norm_intercept": -0.2}


def process_pair(
    raw_path, gt_path, calib, width=1280, height=720, k1=-0.4, max_latency_ms=33.3
):
    df_raw = pd.read_csv(raw_path)
    df_gt = pd.read_csv(gt_path)

    if len(df_raw) != len(df_gt):
        print(
            f"Warning: Row count mismatch in {os.path.basename(raw_path)} ({len(df_raw)} vs {len(df_gt)})."
        )

    df = pd.concat(
        [df_raw.reset_index(drop=True), df_gt.reset_index(drop=True)], axis=1
    )

    mismatches = (df["note"] != df["target_note"]).sum()
    if mismatches > 0:
        print(
            f"Warning: {mismatches} note mismatches in {os.path.basename(raw_path)}."
        )

    initial_count = len(df)
    df["latency_ms"] = (df["midi_time"] - df["video_time"]).abs() * 1000
    df = df[df["latency_ms"] <= max_latency_ms].copy()
    removed_count = initial_count - len(df)

    if removed_count > 0:
        print(
            f"Warning: Removed {removed_count} frame(s) exceeding {max_latency_ms} ms in {os.path.basename(raw_path)}."
        )

    df["note_delta"] = df["target_note"].diff().fillna(0)
    df["norm_pitch"] = (df["target_note"] - 48) / 12.0

    x_cols = [f"x_{i}" for i in range(21)]
    y_cols = [f"y_{i}" for i in range(21)]

    raw_x = df[x_cols].values
    raw_y = df[y_cols].values
    num_samples = len(df)

    pixel_pts = np.stack([raw_x * width, raw_y * height], axis=-1).reshape(
        -1, 1, 2
    )
    focal_length = float(width)
    cx, cy = width / 2.0, height / 2.0
    K = np.array(
        [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]],
        dtype=np.float32,
    )
    D = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    undistorted_pts = cv2.undistortPoints(pixel_pts, K, D, P=K).reshape(
        num_samples, 21, 2
    )

    norm_x = undistorted_pts[:, :, 0] / width
    norm_y = undistorted_pts[:, :, 1] / height

    aspect_ratio = width / float(height)
    x_corrected = norm_x * aspect_ratio
    y_corrected = norm_y

    wrist_x = x_corrected[:, [0]]
    wrist_y = y_corrected[:, [0]]

    rel_x = x_corrected - wrist_x
    rel_y = y_corrected - wrist_y

    hand_scale = np.sqrt(rel_x[:, 9] ** 2 + rel_y[:, 9] ** 2)[:, np.newaxis]
    hand_scale = np.where(hand_scale == 0, 1.0, hand_scale)

    final_x = rel_x / hand_scale
    final_y = rel_y / hand_scale

    clean_df = pd.DataFrame()
    clean_df["target_note"] = df["target_note"].values
    clean_df["norm_pitch"] = df["norm_pitch"].values
    clean_df["note_delta"] = df["note_delta"].values
    clean_df["target_finger"] = df["target_finger"].values

    for i in range(1, 21):
        clean_df[f"dx_{i}"] = final_x[:, i]
        clean_df[f"dy_{i}"] = final_y[:, i]

    fingertip_ids = [4, 8, 12, 16, 20]
    for tip in fingertip_ids:
        clean_df[f"dist_tip_{tip}"] = np.sqrt(
            final_x[:, tip] ** 2 + final_y[:, tip] ** 2
        )

    norm_slope = calib["norm_slope"]
    norm_intercept = calib["norm_intercept"]

    target_notes = df["target_note"].values
    key_norm_x = ((target_notes * norm_slope) + norm_intercept) * aspect_ratio
    wrist_norm_x = wrist_x.flatten()

    rel_key_x = (key_norm_x - wrist_norm_x) / hand_scale.flatten()

    for tip in fingertip_ids:
        clean_df[f"dx_key_tip_{tip}"] = final_x[:, tip] - rel_key_x
        clean_df[f"abs_dx_key_tip_{tip}"] = np.abs(clean_df[f"dx_key_tip_{tip}"])

    clean_df["session_id"] = os.path.basename(raw_path).replace(".csv", "")
    return clean_df


def main():
    calib = load_calibration()
    raw_files = sorted(glob.glob("raw_data/*.csv"))
    if not raw_files:
        print("Error: No raw CSV files found in raw_data/ directory.")
        return

    all_cleaned = []

    for raw_path in raw_files:
        filename = os.path.basename(raw_path).replace(".csv", "")
        gt_key = filename.rsplit("_", 1)[0] if "_" in filename else filename
        gt_path = os.path.join("ground_truths", f"{gt_key}.csv")

        if os.path.exists(gt_path):
            cleaned_sub_df = process_pair(raw_path, gt_path, calib)
            if cleaned_sub_df is not None:
                all_cleaned.append(cleaned_sub_df)
        else:
            print(
                f"Skipping {os.path.basename(raw_path)}: Target ground truth file {gt_path} not found."
            )

    if all_cleaned:
        final_df = pd.concat(all_cleaned, ignore_index=True)
        final_df.to_csv("clean.csv", index=False)
        print(
            f"Processed {len(all_cleaned)} take(s). Saved {len(final_df)} total samples to clean.csv."
        )


if __name__ == "__main__":
    main()