"""
Records user-inputted keys to calibrate horizontal piano key distance.
'q' to quit.
"""

import json
import cv2
import numpy as np

CALIB_NOTES = {"C3": 48, "C4": 60, "C5": 72}
calib_points = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(calib_points) < 3:
        calib_points.append((x, y))
        print(f"Clicked point {len(calib_points)}: ({x}, {y})")


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    cv2.namedWindow("Keyboard Calibration")
    cv2.setMouseCallback("Keyboard Calibration", click_event)

    note_keys = list(CALIB_NOTES.keys())

    print("Click on the middle of the bottom edge of the following keys in order:")
    print("1. C3")
    print("2. C4")
    print("3. C5")
    print("Press 's' to save.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, -1)
        h, w, _ = frame.shape

        current_step = len(calib_points)
        if current_step < 3:
            target_name = note_keys[current_step]
            cv2.putText(
                frame,
                f"Click: {target_name}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Calibration Complete! Press 's' to save.",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        for idx, pt in enumerate(calib_points):
            cv2.circle(frame, pt, 7, (0, 0, 255), -1)
            cv2.putText(
                frame,
                note_keys[idx],
                (pt[0] - 15, pt[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        if len(calib_points) == 3:
            pts_x = [pt[0] for pt in calib_points]
            cv2.line(
                frame,
                (pts_x[0], calib_points[0][1]),
                (pts_x[2], calib_points[2][1]),
                (0, 255, 0),
                2,
            )

        cv2.imshow("Keyboard Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and len(calib_points) == 3:
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    midi_vals = np.array([48, 60, 72], dtype=np.float64)
    pixel_x_vals = np.array(
        [calib_points[0][0], calib_points[1][0], calib_points[2][0]],
        dtype=np.float64,
    )

    slope, intercept = np.polyfit(midi_vals, pixel_x_vals, 1)

    norm_slope = float(slope / 1280.0)
    norm_intercept = float(intercept / 1280.0)

    config = {
        "width": 1280,
        "height": 720,
        "slope_px": float(slope),
        "intercept_px": float(intercept),
        "norm_slope": norm_slope,
        "norm_intercept": norm_intercept,
        "c3_px": calib_points[0],
        "c4_px": calib_points[1],
        "c5_px": calib_points[2],
    }

    with open("calibration_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\nSaved calibration to calibration_config.json:")


if __name__ == "__main__":
    main()