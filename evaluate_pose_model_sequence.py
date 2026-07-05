import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

DATASET_DIR = "dataset_sequence_all"
MODEL_DIR = "models"

X = np.load(os.path.join(DATASET_DIR, "X_csi_seq.npy"))
Y_true = np.load(os.path.join(DATASET_DIR, "Y_pose_seq.npy"))
actions = np.load(os.path.join(DATASET_DIR, "actions_seq.npy"), allow_pickle=True)
sources = np.load(os.path.join(DATASET_DIR, "sources_seq.npy"), allow_pickle=True)

model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "pose_prediction_model_sequence.h5"),
    compile=False
)

scaler = joblib.load(os.path.join(MODEL_DIR, "csi_sequence_scaler.pkl"))

num_samples, seq_len, num_features = X.shape

X_2d = X.reshape(-1, num_features)
X_2d_scaled = scaler.transform(X_2d)
X_scaled = X_2d_scaled.reshape(num_samples, seq_len, num_features)

Y_pred = model.predict(X_scaled)

mae_all = np.mean(np.abs(Y_true - Y_pred))
mse_all = np.mean((Y_true - Y_pred) ** 2)

print("===== 전체 오차 =====")
print("전체 MAE:", mae_all)
print("전체 MSE:", mse_all)

landmark_names = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

joint_rows = []

print("\n===== 관절별 MAE =====")
for i, name in enumerate(landmark_names):
    start = i * 2
    end = start + 2

    joint_mae = np.mean(np.abs(Y_true[:, start:end] - Y_pred[:, start:end]))
    print(f"{name}: {joint_mae:.4f}")

    joint_rows.append({
        "joint": name,
        "mae": joint_mae
    })

action_rows = []

print("\n===== 동작별 MAE =====")
for action in sorted(set(actions)):
    idx = actions == action

    action_mae = np.mean(np.abs(Y_true[idx] - Y_pred[idx]))
    action_mse = np.mean((Y_true[idx] - Y_pred[idx]) ** 2)

    print(f"{action}: MAE={action_mae:.4f}, MSE={action_mse:.4f}, samples={idx.sum()}")

    action_rows.append({
        "action": action,
        "mae": action_mae,
        "mse": action_mse,
        "samples": int(idx.sum())
    })

pd.DataFrame(joint_rows).to_csv(
    "joint_mae_sequence_result.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(action_rows).to_csv(
    "action_mae_sequence_result.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n결과 저장 완료")
print("joint_mae_sequence_result.csv")
print("action_mae_sequence_result.csv")