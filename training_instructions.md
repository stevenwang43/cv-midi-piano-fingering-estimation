# Adding Custom Training Data

This guide outlines the step-by-step workflow for recording, processing, and retraining the piano fingering model on custom hand tracking and MIDI data.

## Step-by-Step Instructions

### Step 1: Define Sequence Ground Truth

Create or specify the sequence target notes and expected fingerings in the `ground_truths/` directory. You can run `0_create_ground_truth.py` or manually construct a CSV file:

```csv
target_note,target_finger
48,1
50,2
52,3
53,1
```

Note: `target_finger` labels map to standard piano fingering (1 = Thumb, 2 = Index, 3 = Middle, 4 = Ring, 5 = Pinky).

### Step 2: Record Synchronized MIDI & Vision Data

Run the data collection script:

```bash
python 1_data_collect.py
```

1. Set `filename = "my_exercise"` near the top of `1_data_collect.py`.
2. Play the matching sequence on your connected USB MIDI keyboard.
3. Keyboard controls during recording:
   - Press **s** to save the take (e.g., `raw_data/my_exercise_1.csv`).
   - Press **d** to discard a take containing playing mistakes.
   - Press **q** to quit.
4. Move complete takes into raw_data/.

### Step 3: Run Feature Extraction & Data Cleaning

Run the cleaning script to align raw landmarks with target ground truth labels:

```bash
python 2_data_clean.py
```

This pipeline performs:
- Lens undistortion using camera parameters.
- Scale and wrist-relative spatial normalization.
- Horizontal key-to-fingertip distance calculations ($\Delta x_{\text{key\_tip}}$).
- Appends clean feature rows to `clean.csv`.

### Step 4: Retrain the Random Forest Classifier

Train an updated classifier across all sessions in `clean.csv`:

```bash
python 3_train_model.py
```

This updates `piano_finger_model.pkl` with your custom dataset weights and outputs Group Cross-Validation metrics.