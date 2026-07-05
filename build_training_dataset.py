import numpy as np
import pandas as pd
from pathlib import Path

DATASET_DIR = Path("carewave_dataset")

MATCHED_DIR = DATASET_DIR / "processed" / "matched"
OUT_DIR = DATASET_DIR / "processed" / "dataset"

OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 10  # 1 second at 10fps
STEP_SIZE = 1
RANDOM_SEED = 42
SPLIT_NAMES = ("train", "val", "test")

CSI_COLS = [
    f"rx{rx}_sub_{i}"
    for rx in [1, 2, 3]
    for i in range(52)
]

POSE_COLS = [
    f"kp_{i}_{axis}"
    for i in range(33)
    for axis in ["x", "y"]
]

VISIBILITY_COLS = [
    f"kp_{i}_visibility"
    for i in range(33)
]

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
MIN_POSE_SCALE = 1e-3


def normalize_pose_sequence(pose_seq: np.ndarray):
    hip_center = (pose_seq[:, LEFT_HIP] + pose_seq[:, RIGHT_HIP]) / 2.0
    shoulder_center = (pose_seq[:, LEFT_SHOULDER] + pose_seq[:, RIGHT_SHOULDER]) / 2.0

    shoulder_width = np.linalg.norm(
        pose_seq[:, LEFT_SHOULDER] - pose_seq[:, RIGHT_SHOULDER],
        axis=1,
    )
    hip_width = np.linalg.norm(
        pose_seq[:, LEFT_HIP] - pose_seq[:, RIGHT_HIP],
        axis=1,
    )
    torso_length = np.linalg.norm(shoulder_center - hip_center, axis=1)

    scale = np.maximum.reduce([shoulder_width, hip_width, torso_length])
    scale = np.maximum(scale, MIN_POSE_SCALE).astype(np.float32)

    normalized = (pose_seq - hip_center[:, None, :]) / scale[:, None, None]

    return normalized.astype(np.float32), hip_center.astype(np.float32), scale[:, None]


def action_from_sample_id(sample_id: str):
    parts = sample_id.split("_")

    if len(parts) <= 2:
        return sample_id

    return "_".join(parts[1:-1])


def make_group_split(info_df: pd.DataFrame):
    split_by_sample = {}
    rng = np.random.default_rng(RANDOM_SEED)

    sample_df = info_df[["sample_id"]].drop_duplicates().copy()
    sample_df["action"] = sample_df["sample_id"].map(action_from_sample_id)

    for _, group in sample_df.groupby("action", sort=True):
        samples = group["sample_id"].to_numpy(dtype=str)
        samples = samples[rng.permutation(len(samples))]

        if len(samples) >= 3:
            split_by_sample[samples[0]] = "test"
            split_by_sample[samples[1]] = "val"
            for sample_id in samples[2:]:
                split_by_sample[sample_id] = "train"
        elif len(samples) == 2:
            split_by_sample[samples[0]] = "val"
            split_by_sample[samples[1]] = "train"
        elif len(samples) == 1:
            split_by_sample[samples[0]] = "train"

    split = info_df["sample_id"].map(split_by_sample).fillna("train").to_numpy(dtype=str)
    split_indices = {
        name: np.flatnonzero(split == name).astype(np.int64)
        for name in SPLIT_NAMES
    }

    return split, split_indices


def build_from_one_file(csv_path: Path):
    sample_id = csv_path.stem.replace("_matched", "")
    df = pd.read_csv(csv_path)

    missing_csi = [c for c in CSI_COLS if c not in df.columns]
    missing_pose = [c for c in POSE_COLS if c not in df.columns]
    missing_visibility = [c for c in VISIBILITY_COLS if c not in df.columns]

    if missing_csi:
        print(f"[SKIP] missing CSI columns: {sample_id}")
        print(missing_csi[:10])
        return [], [], [], [], [], [], []

    if missing_pose:
        print(f"[SKIP] missing pose columns: {sample_id}")
        print(missing_pose[:10])
        return [], [], [], [], [], [], []

    df = df.dropna(subset=CSI_COLS + POSE_COLS).reset_index(drop=True)

    if len(df) < WINDOW_SIZE:
        print(f"[SKIP] not enough rows: {sample_id} / {len(df)} rows")
        return [], [], [], [], [], [], []

    has_time = "time_sec" in df.columns

    X_list = []
    Y_seq_list = []
    Y_norm_seq_list = []
    center_seq_list = []
    scale_seq_list = []
    visibility_seq_list = []
    meta_list = []

    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE

        # Input: 1 second of CSI sequence, shape: (10, 156)
        csi_window = df.loc[start:end - 1, CSI_COLS].values.astype(np.float32)

        # Target: pose sequence for the same 1 second, shape: (10, 33, 2)
        pose_seq = df.loc[start:end - 1, POSE_COLS].values.astype(np.float32)
        pose_seq = pose_seq.reshape(WINDOW_SIZE, 33, 2)
        pose_norm_seq, pose_center_seq, pose_scale_seq = normalize_pose_sequence(pose_seq)

        if missing_visibility:
            visibility_seq = np.ones((WINDOW_SIZE, 33), dtype=np.float32)
        else:
            visibility_seq = df.loc[start:end - 1, VISIBILITY_COLS].values.astype(np.float32)
            visibility_seq = np.nan_to_num(visibility_seq, nan=0.0)

        meta = {
            "sample_id": sample_id,
            "action": action_from_sample_id(sample_id),
            "window_start": start,
            "window_end": end - 1,
        }
        if has_time:
            meta["start_time_sec"] = float(df.loc[start, "time_sec"])
            meta["end_time_sec"] = float(df.loc[end - 1, "time_sec"])

        X_list.append(csi_window)
        Y_seq_list.append(pose_seq)
        Y_norm_seq_list.append(pose_norm_seq)
        center_seq_list.append(pose_center_seq)
        scale_seq_list.append(pose_scale_seq)
        visibility_seq_list.append(visibility_seq)
        meta_list.append(meta)

    print(f"[OK] {sample_id}: {len(X_list)} windows")

    return (
        X_list,
        Y_seq_list,
        Y_norm_seq_list,
        center_seq_list,
        scale_seq_list,
        visibility_seq_list,
        meta_list,
    )


def main():
    matched_files = sorted(MATCHED_DIR.glob("*_matched.csv"))

    print(f"matched file count: {len(matched_files)}")

    all_X = []
    all_Y_seq = []
    all_Y_norm_seq = []
    all_center_seq = []
    all_scale_seq = []
    all_visibility_seq = []
    all_meta = []

    for csv_path in matched_files:
        (
            X,
            Y_seq,
            Y_norm_seq,
            center_seq,
            scale_seq,
            visibility_seq,
            meta,
        ) = build_from_one_file(csv_path)

        all_X.extend(X)
        all_Y_seq.extend(Y_seq)
        all_Y_norm_seq.extend(Y_norm_seq)
        all_center_seq.extend(center_seq)
        all_scale_seq.extend(scale_seq)
        all_visibility_seq.extend(visibility_seq)
        all_meta.extend(meta)

    X_array = np.array(all_X, dtype=np.float32)
    Y_seq_array = np.array(all_Y_seq, dtype=np.float32)
    Y_norm_seq_array = np.array(all_Y_norm_seq, dtype=np.float32)
    center_seq_array = np.array(all_center_seq, dtype=np.float32)
    scale_seq_array = np.array(all_scale_seq, dtype=np.float32)
    visibility_seq_array = np.array(all_visibility_seq, dtype=np.float32)
    info_df = pd.DataFrame(all_meta)
    sample_array = (
        np.array(info_df["sample_id"].astype(str).tolist(), dtype=str)
        if len(info_df)
        else np.array([], dtype=str)
    )

    np.save(OUT_DIR / "X_csi.npy", X_array)
    np.save(OUT_DIR / "Y_pose_seq.npy", Y_seq_array)
    np.save(OUT_DIR / "Y_pose_seq_norm.npy", Y_norm_seq_array)
    np.save(OUT_DIR / "Y_pose_center_seq.npy", center_seq_array)
    np.save(OUT_DIR / "Y_pose_scale_seq.npy", scale_seq_array)
    np.save(OUT_DIR / "Y_visibility_seq.npy", visibility_seq_array)

    # Keep the old single-pose target for backward compatibility.
    Y_last_array = (
        Y_seq_array[:, -1]
        if len(Y_seq_array)
        else np.empty((0, 33, 2), dtype=np.float32)
    )
    Y_norm_last_array = (
        Y_norm_seq_array[:, -1]
        if len(Y_norm_seq_array)
        else np.empty((0, 33, 2), dtype=np.float32)
    )
    visibility_last_array = (
        visibility_seq_array[:, -1]
        if len(visibility_seq_array)
        else np.empty((0, 33), dtype=np.float32)
    )
    np.save(OUT_DIR / "Y_pose.npy", Y_last_array)
    np.save(OUT_DIR / "Y_pose_norm.npy", Y_norm_last_array)
    np.save(OUT_DIR / "Y_visibility.npy", visibility_last_array)

    np.save(OUT_DIR / "sample_ids.npy", sample_array)
    split, split_indices = make_group_split(info_df)
    info_df["split"] = split
    info_df.to_csv(
        OUT_DIR / "dataset_info.csv",
        index=False,
        encoding="utf-8-sig"
    )
    np.savez(
        OUT_DIR / "split_indices.npz",
        train_idx=split_indices["train"],
        val_idx=split_indices["val"],
        test_idx=split_indices["test"],
    )

    split_summary = (
        info_df.groupby("split")["sample_id"]
        .agg(windows="count", samples="nunique")
        .reindex(SPLIT_NAMES, fill_value=0)
    )
    split_summary.to_csv(OUT_DIR / "split_summary.csv", encoding="utf-8-sig")

    print("\nDataset build complete")
    print(f"X_csi shape: {X_array.shape}")
    print(f"Y_pose_seq shape: {Y_seq_array.shape}")
    print(f"Y_pose_seq_norm shape: {Y_norm_seq_array.shape}")
    print(f"Y_pose_center_seq shape: {center_seq_array.shape}")
    print(f"Y_pose_scale_seq shape: {scale_seq_array.shape}")
    print(f"Y_visibility_seq shape: {visibility_seq_array.shape}")
    print(f"Y_pose shape: {Y_last_array.shape}")
    print(f"Y_pose_norm shape: {Y_norm_last_array.shape}")
    print(f"Y_visibility shape: {visibility_last_array.shape}")
    print(f"sample_ids shape: {sample_array.shape}")
    print("split summary:")
    print(split_summary)
    print(f"dataset_info columns: {list(info_df.columns)}")


if __name__ == "__main__":
    main()
