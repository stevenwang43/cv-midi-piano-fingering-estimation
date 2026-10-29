"""
Generates and saves CSV ground-truth files containing target MIDI notes 
and piano fingerings for various scales and arpeggios.
"""

import os
import pandas as pd

os.makedirs("ground_truths", exist_ok=True)

sequences = [
    (
        "c3_maj_scale.csv",
        [48, 50, 52, 53, 55, 57, 59, 60, 59, 57, 55, 53, 52, 50, 48],
        [1, 2, 3, 1, 2, 3, 4, 5, 4, 3, 2, 1, 3, 2, 1],
    ),
    (
        "e3_maj_scale.csv",
        [52, 54, 56, 57, 59, 61, 63, 64, 63, 61, 59, 57, 56, 54, 52],
        [1, 2, 3, 1, 2, 3, 4, 5, 4, 3, 2, 1, 3, 2, 1],
    ),
    (
        "g3_maj_scale.csv",
        [55, 57, 59, 60, 62, 64, 66, 67, 66, 64, 62, 60, 59, 57, 55],
        [1, 2, 3, 1, 2, 3, 4, 5, 4, 3, 2, 1, 3, 2, 1],
    ),
    (
        "c3_hmin_scale.csv",
        [48, 50, 51, 53, 55, 56, 59, 60, 59, 56, 55, 53, 51, 50, 48],
        [1, 2, 3, 1, 2, 3, 4, 5, 4, 3, 2, 1, 3, 2, 1],
    ),
    (
        "f3_hmin_scale.csv",
        [53, 55, 56, 58, 60, 61, 64, 65, 64, 61, 60, 58, 56, 55, 53],
        [1, 2, 3, 4, 1, 2, 3, 4, 3, 2, 1, 4, 3, 2, 1],
    ),
    (
        "c3_arpeggio.csv",
        [48, 52, 55, 60, 64, 67, 72, 67, 64, 60, 55, 52, 48],
        [1, 2, 3, 1, 2, 3, 5, 3, 2, 1, 3, 2, 1],
    ),
    (
        "a2_arpeggio.csv",
        [45, 49, 52, 57, 61, 64, 69, 64, 61, 57, 52, 49, 45],
        [1, 2, 3, 1, 2, 3, 5, 3, 2, 1, 3, 2, 1],
    ),
]


def main():
    for name, notes, fingers in sequences:
        if len(notes) != len(fingers):
            print(f"Error: Mismatch in {name}. Skipping.")
            continue

        df = pd.DataFrame({"target_note": notes, "target_finger": fingers})
        path = os.path.join("ground_truths", name)
        df.to_csv(path, index=False)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()