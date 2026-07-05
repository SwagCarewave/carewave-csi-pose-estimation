import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DATASET_DIR = Path("carewave_dataset")
MODEL_PATH = DATASET_DIR / "models" / "csi_pose_best_bilstm_attention.pth"

OUT_DIR = DATASET_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 10
TIME_BIN = 1 / TARGET_FPS
WIDTH = 640
HEIGHT = 640

SUBCARRIER_COLS = [f"sub_{i}" for i in range(52)]

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

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


class CSIPoseBiLSTMAttention(nn.Module):
    def __init__(self, input_dim=156, hidden_dim=256, num_layers=2, output_dim=132):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        out, _ = self.lstm(x)

        attn_out, _ = self.attn(out, out, out)
        out = self.norm1(out + attn_out)

        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)

        return self.head(out)


def get_sample_id_from_csi(path: Path):
    name = path.stem

    if name.endswith("_csi_raw"):
        return name.replace("_csi_raw", "")

    if name.startswith("csi_raw_"):
        return name.replace("csi_raw_", "csi_")

    return name.replace("_raw", "")


def get_rx_num(rx_id):
    rx_text = str(rx_id).lower().replace("rx", "").strip()
    return int(rx_text)


def read_raw_csi(csv_path: Path):
    fixed_columns = ["experiment_id", "timestamp", "label", "rx"] + SUBCARRIER_COLS

    try:
        df = pd.read_csv(csv_path)

        if "sub_51" not in df.columns:
            df = pd.read_csv(
                csv_path,
                header=None,
                names=fixed_columns,
                skiprows=1,
                engine="python",
                on_bad_lines="skip",
            )

        return df

    except pd.errors.ParserError:
        df = pd.read_csv(
            csv_path,
            header=None,
            names=fixed_columns,
            skiprows=1,
            engine="python",
            on_bad_lines="skip",
        )
        return df


def raw_to_10fps(raw_csv_path: Path):
    sample_id = get_sample_id_from_csi(raw_csv_path)
    df = read_raw_csi(raw_csv_path)

    if "timestamp" not in df.columns:
        raise ValueError("timestamp 컬럼 없음")

    if "rx" not in df.columns:
        raise ValueError("rx 컬럼 없음")

    missing_subs = [c for c in SUBCARRIER_COLS if c not in df.columns]
    if missing_subs:
        raise ValueError(f"subcarrier 컬럼 부족: {missing_subs[:10]}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if df.empty:
        raise ValueError("유효한 timestamp 데이터가 없음")

    start_time = df["timestamp"].min()
    df["time_sec"] = (df["timestamp"] - start_time).dt.total_seconds()
    df["time_bin"] = (df["time_sec"] / TIME_BIN).round().astype(int)

    rows = []

    for time_bin, group in df.groupby("time_bin"):
        row = {
            "sample_id": sample_id,
            "time_sec": time_bin * TIME_BIN,
        }

        for rx_id in sorted(group["rx"].dropna().unique()):
            rx_group = group[group["rx"] == rx_id]
            rx_num = get_rx_num(rx_id)

            for sub in SUBCARRIER_COLS:
                row[f"rx{rx_num}_{sub}"] = rx_group[sub].mean()

        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("time_sec").reset_index(drop=True)

    numeric_cols = out_df.select_dtypes(include=["number"]).columns
    out_df[numeric_cols] = out_df[numeric_cols].interpolate(limit_direction="both")

    return out_df


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = CSIPoseBiLSTMAttention(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        output_dim=checkpoint["output_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_pose_sequence(model, checkpoint, csi, device):
    window_size = int(checkpoint.get("window_size", 10))
    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]

    pred_sum = np.zeros((len(csi), 66), dtype=np.float32)
    pred_count = np.zeros((len(csi), 1), dtype=np.float32)

    for start in range(0, len(csi) - window_size + 1):
        window = csi[start:start + window_size][np.newaxis, :, :]
        window = (window - x_mean) / x_std

        tensor = torch.tensor(window, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred = model(tensor).cpu().numpy()[0]

        # 앞 66개가 raw skeleton 좌표
        pred_raw = pred[:, :66]

        pred_sum[start:start + window_size] += pred_raw
        pred_count[start:start + window_size] += 1.0

    valid = pred_count[:, 0] > 0

    pose_flat = np.zeros((len(csi), 66), dtype=np.float32)
    pose_flat[valid] = pred_sum[valid] / pred_count[valid]
    pose_flat = np.clip(pose_flat, 0.0, 1.0)

    return pose_flat, valid


def draw_pose(points):
    frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    points = np.clip(points.copy(), 0, 1)

    for a, b in POSE_CONNECTIONS:
        x1 = int(points[a, 0] * WIDTH)
        y1 = int(points[a, 1] * HEIGHT)
        x2 = int(points[b, 0] * WIDTH)
        y2 = int(points[b, 1] * HEIGHT)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    for x_norm, y_norm in points:
        x = int(x_norm * WIDTH)
        y = int(y_norm * HEIGHT)
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    return frame


def write_outputs(csi_df, pose_flat, valid, output_prefix):
    out_csv = OUT_DIR / f"{output_prefix}_pred_pose.csv"
    out_video = OUT_DIR / f"{output_prefix}_pred_skeleton.mp4"

    result_df = csi_df[["sample_id", "time_sec"]].copy()
    result_df["valid_prediction"] = valid

    for i, col in enumerate(POSE_COLS):
        result_df[col] = pose_flat[:, i]

    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        TARGET_FPS,
        (WIDTH, HEIGHT),
    )

    for pose_row, is_valid in zip(pose_flat, valid):
        if is_valid:
            frame = draw_pose(pose_row.reshape(33, 2))
        else:
            frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

        writer.write(frame)

    writer.release()

    return out_csv, out_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-csi",
        type=Path,
        required=True,
        help="테스트할 CSI raw csv 경로",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="출력 파일명 prefix",
    )
    args = parser.parse_args()

    output_prefix = args.output_prefix or get_sample_id_from_csi(args.raw_csi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model, checkpoint = load_model(device)

    print("model:", MODEL_PATH)
    print("best_epoch:", checkpoint.get("best_epoch"))
    print("best_val_loss:", checkpoint.get("best_val_loss"))

    csi_df = raw_to_10fps(args.raw_csi)
    print("10fps CSI shape:", csi_df.shape)

    missing_cols = [c for c in CSI_COLS if c not in csi_df.columns]
    if missing_cols:
        raise ValueError(f"CSI 컬럼 부족: {missing_cols[:10]}")

    csi = csi_df[CSI_COLS].values.astype(np.float32)

    pose_flat, valid = predict_pose_sequence(model, checkpoint, csi, device)

    out_csv, out_video = write_outputs(csi_df, pose_flat, valid, output_prefix)

    print("saved csv:", out_csv)
    print("saved video:", out_video)
    print("valid frames:", int(valid.sum()), "/", len(valid))


if __name__ == "__main__":
    main()