"""
Runs a real-time visualization and evaluation loop that captures webcam video, 
tracks hand joints using MediaPipe, synchronizes incoming MIDI note events, and 
renders an on-screen HUD to evaluate predicted piano finger accuracy against presets.
'q' to quit.
"""

import json
import os
import queue
import sys
import time
import cv2
import joblib
import mido
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

curr_landmarks = None
midi_queue = queue.Queue()

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
FINGER_IDS = [4, 8, 12, 16, 20]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

FINGER_SEQUENCES = {
    "1": {
        "name": "1-Octave Scale, turn on 3",
        "sequence": [1, 2, 3, 1, 2, 3, 4, 5, 4, 3, 2, 1, 3, 2, 1],
    },
    "2": {
            "name": "Fur Elise",
            "sequence": [5, 4, 5, 4, 5, 2, 4, 3, 1, 1, 2, 4, 5, 1, 2, 3, 4, 1, 5, 4, 5, 4, 5, 2, 4, 3, 1, 1, 2, 4, 5, 1, 3, 2, 1]
    },
    "3": {
        "name": "Pathetique 3rd Mvt.",
        "sequence": [1, 2, 3, 4, 5, 3, 4, 2, 
                     3, 2, 1, 2, 1, 3, 2, 1, 2, 3, 4, 4, 4, 
                     2, 3, 2, 3, 4, 1, 2, 3, 2, 3, 4, 1, 
                     1, 2, 3, 3, 4, 2, 2, 3, 1, 
                     3, 4, 5, 2, 3, 4, 5, 2, 2, 3, 4, 4, 5, 2, 2, 3, 1, 
                     4, 4, 3, 2, 1, 3, 2, 1, 2, 1, 
                     5, 4, 3, 2, 1, 3, 2, 1, 2, 1, 2, 3, 2, 1, 2, 3, 5, 5],
    },
}

pred_history = []
match_history = []
latest_confidences = {}


def load_calibration():
    if os.path.exists("calibration_config.json"):
        with open("calibration_config.json", "r") as f:
            return json.load(f)
    print("Warning: Using default calibration.")
    return {"norm_slope": 0.015, "norm_intercept": -0.2}


def midi_to_note_name(midi_num):
    return f"{NOTE_NAMES[midi_num % 12]}{(midi_num // 12) - 1}"


def on_midi_input(msg):
    if msg.type == "note_on" and msg.velocity > 0:
        midi_queue.put({"note": msg.note, "midi_time": time.perf_counter()})


def on_hand_result(result, image, time_ms):
    global curr_landmarks
    curr_landmarks = result.hand_landmarks[0] if result.hand_landmarks else None


def extract_live_features(landmarks_2d, current_note, prev_note, calib, width=1280, height=720, k1=-0.4):
    pixel_pts = (landmarks_2d * np.array([width, height], dtype=np.float32)).reshape(-1, 1, 2)
    focal_length = float(width)
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32)
    D = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    undistorted_pts = cv2.undistortPoints(pixel_pts, K, D, P=K).reshape(21, 2)
    norm_x = undistorted_pts[:, 0] / width
    norm_y = undistorted_pts[:, 1] / height

    aspect_ratio = width / float(height)
    x_corrected = norm_x * aspect_ratio
    y_corrected = norm_y

    wrist_x, wrist_y = x_corrected[0], y_corrected[0]
    rel_x = x_corrected - wrist_x
    rel_y = y_corrected - wrist_y

    hand_scale = np.sqrt(rel_x[9] ** 2 + rel_y[9] ** 2)
    if hand_scale == 0:
        hand_scale = 1.0

    final_x = rel_x / hand_scale
    final_y = rel_y / hand_scale

    feat = {}
    feat["norm_pitch"] = (current_note - 48) / 12.0
    feat["note_delta"] = float(current_note - prev_note) if prev_note is not None else 0.0

    for i in range(1, 21):
        feat[f"dx_{i}"] = final_x[i]
        feat[f"dy_{i}"] = final_y[i]

    for id in FINGER_IDS:
        feat[f"dist_tip_{id}"] = np.sqrt(final_x[id] ** 2 + final_y[id] ** 2)

    norm_slope = calib["norm_slope"]
    norm_intercept = calib["norm_intercept"]

    key_norm_x = ((current_note * norm_slope) + norm_intercept) * aspect_ratio
    rel_key_x = (key_norm_x - wrist_x) / hand_scale

    for id in FINGER_IDS:
        feat[f"dx_key_tip_{id}"] = final_x[id] - rel_key_x
        feat[f"abs_dx_key_tip_{id}"] = abs(feat[f"dx_key_tip_{id}"])

    return pd.DataFrame([feat])


def draw_hud(img, active_note_str, active_note_num, total_notes, total_correct, pattern_name, base_sequence, total_events_played, confidences):
    overlay = img.copy()

    panel_w, panel_h = 520, 205
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (10, 10), (10 + panel_w, 10 + panel_h), (255, 255, 255), 1)

    live_acc = (total_correct / total_notes * 100.0) if total_notes > 0 else 100.0
    acc_color = (0, 255, 0) if live_acc >= 85.0 else (0, 165, 255) if live_acc >= 70.0 else (0, 0, 255)

    cv2.putText(img, f"Song: {pattern_name}", (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(img, f"Note: {active_note_str} ({active_note_num})", (25, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(img, f"Acc: {live_acc:.1f}% ({total_correct}/{total_notes})", (300, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, acc_color, 2)

    cv2.line(img, (20, 70), (panel_w, 70), (100, 100, 100), 1)

    window_size = 9
    center_offset = 4
    seq_len = len(base_sequence)

    start_event_idx = max(0, total_events_played - center_offset)

    start_x = 90
    spacing = 42

    active_slot = total_events_played - start_event_idx
    if 0 <= active_slot < window_size:
        active_slot_x = start_x + (active_slot * spacing)
        cv2.rectangle(img, (active_slot_x - 6, 82), (active_slot_x + 20, 148), (0, 255, 255), 1)

    cv2.putText(img, "True:", (25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    for i in range(window_size):
        event_i = start_event_idx + i
        x_pos = start_x + (i * spacing)

        if event_i > 0 and (event_i % seq_len == 0):
            cv2.putText(img, "|", (x_pos - 18, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(img, "|", (x_pos - 18, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        exp_val = base_sequence[event_i % seq_len]
        color = (255, 255, 255) if event_i >= total_events_played else (150, 150, 150)
        cv2.putText(img, str(exp_val), (x_pos, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(img, "Pred:", (25, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    for i in range(window_size):
        event_i = start_event_idx + i
        x_pos = start_x + (i * spacing)
        if event_i < len(pred_history):
            pred_val = pred_history[event_i]
            is_match = match_history[event_i]
            color = (0, 255, 0) if is_match else (0, 0, 255)
            cv2.putText(img, str(pred_val), (x_pos, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        else:
            cv2.putText(img, "-", (x_pos, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

    cv2.line(img, (20, 155), (panel_w, 155), (100, 100, 100), 1)

    cv2.putText(img, "Proba:", (25, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    sorted_fingers = sorted(confidences.keys(), key=lambda k: confidences[k], reverse=True)
    rank_1 = sorted_fingers[0] if len(sorted_fingers) > 0 else None
    rank_2 = sorted_fingers[1] if len(sorted_fingers) > 1 else None

    is_close_second = False
    if rank_1 is not None and rank_2 is not None:
        prob_diff = confidences[rank_1] - confidences[rank_2]
        if prob_diff <= 0.2:
            is_close_second = True

    prob_x = 90
    for finger_id in [1, 2, 3, 4, 5]:
        prob_val = confidences.get(finger_id, 0.0)
        prob_str = f"F{finger_id}:{prob_val:.2f}"
        
        if finger_id == rank_1:
            p_color = (0, 255, 0)
        elif finger_id == rank_2 and is_close_second:
            p_color = (0, 255, 255)
        else:
            p_color = (150, 150, 150)
        
        cv2.putText(img, prob_str, (prob_x, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.4, p_color, 1)
        prob_x += 84


def draw_skeleton(img, landmarks, width, height):
    pts = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    for p1, p2 in HAND_CONNECTIONS:
        cv2.line(img, pts[p1], pts[p2], (255, 255, 255), 1)

    for idx, pt in enumerate(pts):
        color = (0, 0, 255) if idx in FINGER_IDS else (0, 255, 0)
        radius = 3 if idx in FINGER_IDS else 2
        cv2.circle(img, pt, radius, color, -1)


def main():
    global latest_confidences

    print("\nSelect a target finger sequence:")
    print("   [1] 1-Octave Scale, turn on 3")
    print("   [2] Fur Elise")
    print("   [3] Pathetique 3rd Mvt.")
    choice = input("Choice (1-3): ").strip()
    seq_data = FINGER_SEQUENCES.get(choice, FINGER_SEQUENCES["1"])
    base_sequence = seq_data["sequence"]
    seq_len = len(base_sequence)

    calib = load_calibration()
    try:
        model = joblib.load("piano_finger_model.pkl")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    midi_inputs = mido.get_input_names()
    if not midi_inputs:
        print("Error: No MIDI device found.")
        sys.exit(1)

    inport = mido.open_input(midi_inputs[0], callback=on_midi_input)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        result_callback=on_hand_result,
    )

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    prev_note = None
    last_note_str = "---"
    last_note_num = 0

    total_events_played = 0
    total_notes = 0
    total_correct = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            img = cv2.flip(img, -1)
            h, w, _ = img.shape

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_img, int(time.perf_counter() * 1000))

            while not midi_queue.empty():
                event = midi_queue.get()
                current_note = event["note"]
                note_str = midi_to_note_name(current_note)

                if curr_landmarks:
                    landmarks_2d = np.array([[lm.x, lm.y] for lm in curr_landmarks])
                    X_live = extract_live_features(landmarks_2d, current_note, prev_note, calib, w, h)

                    probs = model.predict_proba(X_live)[0]
                    all_confidences = {int(cls): round(float(prob), 4) for cls, prob in zip(model.classes_, probs)}
                    latest_confidences = all_confidences

                    ml_finger = int(model.classes_[np.argmax(probs)])

                    expected_finger = base_sequence[total_events_played % seq_len]
                    total_notes += 1
                    is_correct = (ml_finger == expected_finger)
                    if is_correct:
                        total_correct += 1

                    pred_history.append(ml_finger)
                    match_history.append(is_correct)
                    total_events_played += 1

                    last_note_str = note_str
                    last_note_num = current_note

                    prev_note = current_note

            if curr_landmarks:
                draw_skeleton(img, curr_landmarks, w, h)

            draw_hud(
                img,
                last_note_str,
                last_note_num,
                total_notes,
                total_correct,
                seq_data["name"],
                base_sequence,
                total_events_played,
                latest_confidences,
            )

            cv2.imshow("piano", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    if inport:
        inport.close()


if __name__ == "__main__":
    main()