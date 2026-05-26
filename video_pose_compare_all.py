import os
import glob
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf

# =========================
# 1. 경로 설정
# =========================
MODEL_PATH = "models/pose_prediction_model_sequence.h5"
SCALER_PATH = "models/csi_sequence_scaler.pkl"

X_PATH = "dataset_sequence_all/X_csi_seq.npy"
Y_PATH = "dataset_sequence_all/Y_pose_seq.npy"
TIMES_PATH = "dataset_sequence_all/times_seq.npy"
SOURCES_PATH = "dataset_sequence_all/sources_seq.npy"
ACTIONS_PATH = "dataset_sequence_all/actions_seq.npy"

RAW_DIR = "dataset_raw"

# =========================
# 2. 데이터 로드
# =========================
X = np.load(X_PATH)
Y_true = np.load(Y_PATH)
times = np.load(TIMES_PATH)
sources = np.load(SOURCES_PATH, allow_pickle=True)
actions = np.load(ACTIONS_PATH, allow_pickle=True)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# =========================
# 3. Sequence 데이터 스케일링
# =========================
# X shape: (samples, seq_len, features)
num_samples, seq_len, num_features = X.shape

X_2d = X.reshape(-1, num_features)
X_2d_scaled = scaler.transform(X_2d)
X_scaled = X_2d_scaled.reshape(num_samples, seq_len, num_features)

Y_pred = model.predict(X_scaled)

# =========================
# 4. 예측 결과 smoothing
# =========================
def smooth_predictions(pred, alpha=0.7):
    smoothed = pred.copy()

    for i in range(1, len(pred)):
        # 같은 영상(source) 안에서만 smoothing
        if sources[i] == sources[i - 1]:
            smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * pred[i]
        else:
            smoothed[i] = pred[i]

    return smoothed

Y_pred = smooth_predictions(Y_pred, alpha=0.7)

print("X shape:", X.shape)
print("X_scaled shape:", X_scaled.shape)
print("Y_true shape:", Y_true.shape)
print("Y_pred shape:", Y_pred.shape)
print("times shape:", times.shape)
print("sources shape:", sources.shape)
print("actions shape:", actions.shape)

# =========================
# 5. 영상별 인덱스 범위 출력
# =========================
print("\n===== 영상별 sample_idx 범위 =====")
for src in sorted(set(sources)):
    idx = np.where(sources == src)[0]
    print(f"{src}: {idx[0]} ~ {idx[-1]} / samples={len(idx)} / action={actions[idx[0]]}")

# =========================
# 6. 오차 계산
# =========================
mae_all = np.mean(np.abs(Y_true - Y_pred))
mse_all = np.mean((Y_true - Y_pred) ** 2)

print("\n===== 전체 오차 =====")
print("전체 MAE:", mae_all)
print("전체 MSE:", mse_all)

# =========================
# 7. 스켈레톤 연결 정보
# =========================
connections = [
    (1, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8),
    (7, 9), (9, 11),
    (8, 10), (10, 12),
    (0, 1), (0, 2)
]

def reshape_pose(flat_pose):
    return flat_pose.reshape(-1, 2)

def draw_skeleton(ax, pose, title):
    ax.clear()

    xs = pose[:, 0]
    ys = pose[:, 1]

    ax.scatter(xs, ys)

    for start, end in connections:
        ax.plot(
            [pose[start, 0], pose[end, 0]],
            [pose[start, 1], pose[end, 1]]
        )

    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

def find_video_path(source_name):
    matches = glob.glob(
        os.path.join(RAW_DIR, "**", f"{source_name}.mp4"),
        recursive=True
    )

    if len(matches) == 0:
        print("[ERROR] 영상 파일 못 찾음:", source_name)
        return None

    return matches[0]

def get_video_frame_by_source(source_name, video_time_sec):
    video_path = find_video_path(source_name)

    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] 영상 열기 실패:", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_number = int(video_time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# =========================
# 8. 보고 싶은 구간 설정
# =========================
# 실행하면 위에 영상별 idx 범위가 출력됨.
# 보고 싶은 영상 범위로 아래 값을 바꾸면 됨.
START_IDX = 0
END_IDX = 570

# 3칸씩 건너뜀 = 약 0.6초 간격
frame_indices = list(range(START_IDX, min(END_IDX, len(X)), 3))

# =========================
# 9. 화면 구성
# =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

def update(sample_idx):
    csi_time = times[sample_idx]
    video_time = csi_time

    source_name = sources[sample_idx]
    action_name = actions[sample_idx]

    frame = get_video_frame_by_source(source_name, video_time)

    true_pose = reshape_pose(Y_true[sample_idx])
    pred_pose = reshape_pose(Y_pred[sample_idx])

    # 1) 실제 영상
    axes[0].clear()

    if frame is not None:
        axes[0].imshow(frame)

    axes[0].set_title(
        f"Original Video\n{source_name}\nAction: {action_name}\nTime: {video_time:.1f}s"
    )
    axes[0].axis("off")

    # 2) 실제 MediaPipe Pose
    draw_skeleton(
        axes[1],
        true_pose,
        f"Real Pose from MediaPipe\n{source_name}\nCSI Time: {csi_time:.1f}s"
    )

    # 3) CSI 예측 Pose
    draw_skeleton(
        axes[2],
        pred_pose,
        f"Predicted Pose from CSI\nSample idx: {sample_idx}"
    )

    return axes

ani = FuncAnimation(
    fig,
    update,
    frames=frame_indices,
    interval=100,
    repeat=True
)

plt.tight_layout()
plt.show()