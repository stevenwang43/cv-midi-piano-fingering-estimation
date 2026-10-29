"""
Matches user-recorded MediaPipe hand-tracking landmarks with MIDI input to create training data.
Features: video_time,midi_time,note,x_0,y_0,x_1,y_1,...,x_20,y_20.
'q' to quit.
"""

import cv2
import time
import mido
import queue
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

filename = "f3_hmin_scale"
midi_queue = queue.Queue()
curr_landmarks = None

def on_midi_input(msg):
    if msg.type == "note_on" and msg.velocity > 0:
        midi_queue.put({
            "note" : msg.note,
            "midi_time": time.perf_counter()
            })

def on_hand_result(result, image, time_ms):
    global curr_landmarks
    if result.hand_landmarks:
        curr_landmarks = result.hand_landmarks[0]

def main():
    take = 1
    midi_inputs = mido.get_input_names()
    if midi_inputs:
        inport = mido.open_input(midi_inputs[0], callback=on_midi_input)
        print(f"Listening to MIDI callback on: {inport.name}")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        result_callback=on_hand_result
        )

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    output = []
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Failed to read frame")
                break

            frame_time = time.perf_counter()
            img = cv2.flip(img, -1)
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            time_ms = int(time.perf_counter() * 1000)
            landmarker.detect_async(mp_img, time_ms)
            
            if curr_landmarks:
                for lm in curr_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

            while not midi_queue.empty():
                event = midi_queue.get()
                if curr_landmarks:
                    frame = {}
                    frame["video_time"] = frame_time
                    frame["midi_time"] = event["midi_time"]
                    frame["note"] = event["note"]
                    print(f"Note {frame["note"]} down at {frame["video_time"]:.3f}s midi, {frame["midi_time"]:.3f}s video")
                    for i, lm in enumerate(curr_landmarks):
                        frame[f"x_{i}"] = lm.x
                        frame[f"y_{i}"] = lm.y
                    output.append(frame)

            cv2.imshow("image", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                df = pd.DataFrame(output)
                df.to_csv(f"{filename}_{take}.csv", index=False)
                print(f"Take {take} ({len(output)} notes)")
                take += 1
                output.clear()
            elif key == ord("d"):
                print(f"Take cleared ({len(output)} notes)")
                output.clear()

        cap.release()
        cv2.destroyAllWindows()
        if inport:
            inport.close()

if __name__ == "__main__":
    main()