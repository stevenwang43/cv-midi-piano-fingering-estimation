"""
Trains a Random Forest classifier using group 
cross-validation to predict the target piano finger.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold


def main():
    df = pd.read_csv("clean.csv")

    y = df["target_finger"]
    groups = df["session_id"]

    X = df.drop(columns=["target_finger", "session_id", "target_note"])

    n_splits = min(5, len(groups.unique()))
    gkf = GroupKFold(n_splits=n_splits)

    accuracies = []
    oof_preds = np.zeros(len(df))

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        oof_preds[test_idx] = y_pred
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Fold {fold} Accuracy: {acc * 100:.2f}%")

    print(f"\nMean Group CV Accuracy: {np.mean(accuracies) * 100:.2f}%\n")
    print(classification_report(y, oof_preds, digits=4))

    output = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    output.fit(X, y)
    joblib.dump(output, "piano_finger_model.pkl")
    print("Saved production model to 'piano_finger_model.pkl'")


if __name__ == "__main__":
    main()