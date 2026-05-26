# build_sequence_dataset_all.py

import os
import numpy as np

DATASET_DIR = "dataset_all"
OUTPUT_DIR = "dataset_sequence_all"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 15   # 15개 샘플 = 약 3초 흐름

X = np.load(os.path.join(DATASET_DIR, "X_csi_all.npy"))
Y = np.load(os.path.join(DATASET_DIR, "Y_pose_all.npy"))
times = np.load(os.path.join(DATASET_DIR, "times_all.npy"))
actions = np.load(os.path.join(DATASET_DIR, "actions_all.npy"), allow_pickle=True)
sources = np.load(os.path.join(DATASET_DIR, "sources_all.npy"), allow_pickle=True)

X_seq = []
Y_seq = []
times_seq = []
actions_seq = []
sources_seq = []

for src in sorted(set(sources)):
    idx = np.where(sources == src)[0]

    # 영상 하나 안에서만 sequence 생성
    for i in range(len(idx) - SEQ_LEN + 1):
        seq_idx = idx[i:i + SEQ_LEN]

        # 마지막 시점의 pose를 정답으로 사용
        target_idx = seq_idx[-1]

        X_seq.append(X[seq_idx])
        Y_seq.append(Y[target_idx])
        times_seq.append(times[target_idx])
        actions_seq.append(actions[target_idx])
        sources_seq.append(sources[target_idx])

X_seq = np.array(X_seq, dtype=np.float32)
Y_seq = np.array(Y_seq, dtype=np.float32)
times_seq = np.array(times_seq, dtype=np.float32)
actions_seq = np.array(actions_seq)
sources_seq = np.array(sources_seq)

np.save(os.path.join(OUTPUT_DIR, "X_csi_seq.npy"), X_seq)
np.save(os.path.join(OUTPUT_DIR, "Y_pose_seq.npy"), Y_seq)
np.save(os.path.join(OUTPUT_DIR, "times_seq.npy"), times_seq)
np.save(os.path.join(OUTPUT_DIR, "actions_seq.npy"), actions_seq)
np.save(os.path.join(OUTPUT_DIR, "sources_seq.npy"), sources_seq)

print("시퀀스 데이터셋 생성 완료")
print("X_seq shape:", X_seq.shape)
print("Y_seq shape:", Y_seq.shape)
print("times_seq shape:", times_seq.shape)
print("actions_seq shape:", actions_seq.shape)
print("sources_seq shape:", sources_seq.shape)