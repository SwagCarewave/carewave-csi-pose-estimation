#!/usr/bin/env python3
"""CareWave raw CSI -> pose -> fall/non-fall inference (fixed version).

Usage:
    python predict_fall.py path/to/csi_raw.csv

Expected files (in ./models or next to this script):
    carewave_315_best.pt
    csi_robust_scaler_train_only.npz
    fall_bigru_attention_best.pt
    fall_feature_normalization.npz
    fall_decision_config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


MODEL_FILES = (
    "carewave_315_best.pt",
    "csi_robust_scaler_train_only.npz",
    "fall_bigru_attention_best.pt",
    "fall_feature_normalization.npz",
    "fall_decision_config.json",
)
RX_ORDER = ("RX1", "RX2", "RX3")
SUB_COLS = [f"sub_{i}" for i in range(52)]


class ReceiverEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class CareWavePoseModel(nn.Module):
    def __init__(
        self,
        pose_prior: np.ndarray,
        center_prior: np.ndarray,
        log_scale_prior: np.ndarray,
        lstm_hidden: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.receiver_encoder = ReceiverEncoder()
        self.frame_projection = nn.Sequential(
            nn.Linear(195, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.temporal_model = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attention = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.context_network = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pose_head = nn.Linear(256, 66)
        self.center_head = nn.Linear(256, 2)
        self.log_scale_head = nn.Linear(256, 1)
        # 학습 노트북과 동일하게 prior는 출력 head의 초기 bias로만 사용됩니다.
        with torch.no_grad():
            nn.init.zeros_(self.pose_head.weight)
            nn.init.zeros_(self.center_head.weight)
            nn.init.zeros_(self.log_scale_head.weight)
            self.pose_head.bias.copy_(torch.as_tensor(pose_prior, dtype=torch.float32))
            self.center_head.bias.copy_(torch.as_tensor(center_prior, dtype=torch.float32))
            self.log_scale_head.bias.copy_(
                torch.as_tensor(log_scale_prior, dtype=torch.float32).reshape(1)
            )

    def forward(self, x: torch.Tensor):
        batch, steps, _ = x.shape
        centered = x[:, :, :156].reshape(batch, steps, 3, 52)
        difference = x[:, :, 156:312].reshape(batch, steps, 3, 52)
        medians = x[:, :, 312:315]

        receiver_embeddings = []
        for rx_index in range(3):
            rx = torch.stack(
                (centered[:, :, rx_index], difference[:, :, rx_index]), dim=2
            ).reshape(batch * steps, 2, 52)
            encoded = self.receiver_encoder(rx).reshape(batch, steps, 64)
            receiver_embeddings.append(encoded)

        frames = torch.cat((*receiver_embeddings, medians), dim=-1)
        frames = self.frame_projection(frames)
        temporal, _ = self.temporal_model(frames)
        attention = torch.softmax(self.attention(temporal).squeeze(-1), dim=1)
        context = torch.sum(temporal * attention.unsqueeze(-1), dim=1)
        context = self.context_network(context)

        pose = self.pose_head(context)
        center = self.center_head(context)
        log_scale = self.log_scale_head(context)
        return pose, center, log_scale, attention


class FallBiGRUAttention(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        input_dim = int(config.get("input_dim", 138))
        projection = int(config.get("projection_dim", 128))
        hidden = int(config.get("hidden_dim", 96))
        layers = int(config.get("gru_layers", 2))
        dropout = float(config.get("dropout", 0.3))

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, projection),
            nn.LayerNorm(projection),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_model = nn.GRU(
            input_size=projection,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        temporal_dim = hidden * 2
        self.attention = nn.Sequential(
            nn.Linear(temporal_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor):
        x = self.input_projection(x)
        temporal, _ = self.temporal_model(x)
        attention = torch.softmax(self.attention(temporal).squeeze(-1), dim=1)
        context = torch.sum(temporal * attention.unsqueeze(-1), dim=1)
        return self.classifier(context).squeeze(-1), attention


def locate_models(model_dir: Path | None) -> dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    candidates = [model_dir] if model_dir else [script_dir / "models", script_dir]
    for directory in candidates:
        if directory and all((directory / name).is_file() for name in MODEL_FILES):
            return {name: directory / name for name in MODEL_FILES}
    checked = ", ".join(str(p) for p in candidates if p)
    raise FileNotFoundError(
        f"모델 파일 5개를 찾지 못했습니다. 확인한 폴더: {checked}\n"
        f"필요한 파일: {', '.join(MODEL_FILES)}"
    )


def read_raw_csi(path: Path) -> pd.DataFrame:
    columns = ["experiment_id", "timestamp", "label", "rx", *SUB_COLS]
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        if first and not (first[0].strip().lower() == "experiment_id"):
            reader = iter([first, *reader])
        for row in reader:
            if len(row) < 4:
                continue
            row = row[:56] + [np.nan] * max(0, 56 - len(row))
            rows.append(row)
    if not rows:
        raise ValueError("CSV에서 유효한 CSI 행을 찾지 못했습니다.")
    frame = pd.DataFrame(rows, columns=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["rx"] = frame["rx"].astype(str).str.strip().str.upper()
    frame[SUB_COLS] = frame[SUB_COLS].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame = frame[frame["rx"].isin(RX_ORDER)].copy()
    if frame.empty:
        raise ValueError("RX1/RX2/RX3 CSI 행을 찾지 못했습니다.")

    # 학습 데이터 생성 당시와 동일한 결측치 보간 순서:
    # 수신기별 시간 방향 -> 같은 행의 서브캐리어 방향 -> 전체 시간 정렬
    frame = frame.sort_values(["rx", "timestamp"]).reset_index(drop=True)
    filled_groups = []
    for _, rx_frame in frame.groupby("rx", sort=False):
        rx_frame = rx_frame.copy()
        rx_frame[SUB_COLS] = rx_frame[SUB_COLS].interpolate(
            method="linear", axis=0, limit_direction="both"
        )
        filled_groups.append(rx_frame)
    frame = pd.concat(filled_groups, ignore_index=True)
    frame[SUB_COLS] = frame[SUB_COLS].interpolate(
        method="linear", axis=1, limit_direction="both"
    )
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def resample_csi_at_10hz(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """학습 노트북과 동일하게 CSI를 0.1초 bin으로 평균·보간합니다."""
    start_timestamp = frame["timestamp"].min()
    frame = frame.copy()
    frame["time_sec_raw"] = (
        frame["timestamp"] - start_timestamp
    ).dt.total_seconds()
    frame["time_bin"] = np.rint(frame["time_sec_raw"] * 10).astype(int)

    grouped = frame.groupby(["time_bin", "rx"], as_index=False)[SUB_COLS].mean()
    minimum_bin = int(grouped["time_bin"].min())
    maximum_bin = int(grouped["time_bin"].max())
    full_bins = np.arange(minimum_bin, maximum_bin + 1, dtype=int)

    receiver_arrays = []
    for rx_name in RX_ORDER:
        rx_values = (
            grouped[grouped["rx"] == rx_name]
            .set_index("time_bin")
            .reindex(full_bins)[SUB_COLS]
            .astype(float)
        )
        rx_values = rx_values.interpolate(
            method="linear", axis=0, limit_direction="both"
        )
        rx_values = rx_values.interpolate(
            method="linear", axis=1, limit_direction="both"
        )
        if rx_values.isna().any().any():
            missing = int(rx_values.isna().sum().sum())
            raise ValueError(f"{rx_name}: 10Hz 변환 후 결측값 {missing}개가 남았습니다.")
        receiver_arrays.append(rx_values.to_numpy(dtype=np.float32))

    signals = np.stack(receiver_arrays, axis=1).astype(np.float32)
    times_sec = (full_bins - minimum_bin).astype(np.float64) / 10.0
    if len(signals) < 20:
        raise ValueError(f"10Hz CSI 프레임이 {len(signals)}개뿐입니다. 최소 20개가 필요합니다.")
    return signals, times_sec


def make_features_315(signals: np.ndarray) -> np.ndarray:
    medians = np.median(signals, axis=2)
    centered = signals - medians[:, :, None]
    difference = np.zeros_like(centered)
    difference[1:] = np.abs(centered[1:] - centered[:-1])
    return np.concatenate(
        (centered.reshape(len(signals), 156), difference.reshape(len(signals), 156), medians),
        axis=1,
    ).astype(np.float32)


def make_pose_sequences(features: np.ndarray, times: np.ndarray):
    windows, target_times = [], []
    for start in range(0, len(features) - 20 + 1, 5):
        end = start + 20
        gaps = np.diff(times[start:end])
        if np.any(gaps <= 0) or np.any(gaps > 0.15):
            continue
        windows.append(features[start:end])
        target_times.append(times[end - 1])
    if not windows:
        raise ValueError("20-frame 시퀀스를 만들지 못했습니다. timestamp 단절 여부를 확인하세요.")
    return np.stack(windows).astype(np.float32), np.asarray(target_times, np.float32)


def batched_pose_inference(model, x, device, batch_size=128):
    pose, center, log_scale = [], [], []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            p, c, s, _ = model(batch)
            pose.append(p.cpu().numpy())
            center.append(c.cpu().numpy())
            log_scale.append(s.cpu().numpy())
    return np.concatenate(pose), np.concatenate(center), np.concatenate(log_scale)


def predict(csv_path: Path, model_dir: Path | None, output_dir: Path | None, cpu: bool = False):
    paths = locate_models(model_dir)
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")

    raw = read_raw_csi(csv_path)
    signals, frame_times = resample_csi_at_10hz(raw)
    features = make_features_315(signals)
    csi_scaler = np.load(paths["csi_robust_scaler_train_only.npz"])
    iqr = np.where(np.abs(csi_scaler["iqr"]) < 1e-8, 1.0, csi_scaler["iqr"])
    features = ((features - csi_scaler["median"]) / iqr).astype(np.float32)
    x_pose, pose_times = make_pose_sequences(features, frame_times)

    pose_ckpt = torch.load(paths["carewave_315_best.pt"], map_location="cpu", weights_only=False)
    pose_model = CareWavePoseModel(
        pose_ckpt["pose_prior"], pose_ckpt["center_prior"], pose_ckpt["log_scale_prior"]
    )
    pose_model.load_state_dict(pose_ckpt["model_state_dict"], strict=True)
    pose_model.to(device).eval()
    pose, center, log_scale = batched_pose_inference(pose_model, x_pose, device)

    base69 = np.concatenate((pose, center, log_scale), axis=1).astype(np.float32)
    delta69 = np.zeros_like(base69)
    delta69[1:] = base69[1:] - base69[:-1]
    fall_features = np.concatenate((base69, delta69), axis=1)
    if len(fall_features) < 6:
        raise ValueError(f"Pose 출력이 {len(fall_features)}개뿐입니다. 2차 모델에는 최소 6개가 필요합니다.")
    fall_windows = np.stack(
        [fall_features[i:i + 6] for i in range(len(fall_features) - 5)]
    ).astype(np.float32)

    fall_norm = np.load(paths["fall_feature_normalization.npz"])
    fall_iqr = np.where(np.abs(fall_norm["feature_iqr"]) < 1e-8, 1.0, fall_norm["feature_iqr"])
    fall_windows = (fall_windows - fall_norm["feature_median"]) / fall_iqr
    fall_windows = np.clip(fall_windows, float(fall_norm["clip_min"]), float(fall_norm["clip_max"])).astype(np.float32)

    fall_ckpt = torch.load(paths["fall_bigru_attention_best.pt"], map_location="cpu", weights_only=False)
    fall_model = FallBiGRUAttention(fall_ckpt.get("model_config", {}))
    fall_model.load_state_dict(fall_ckpt["model_state_dict"], strict=True)
    fall_model.to(device).eval()
    with torch.inference_mode():
        logits, _ = fall_model(torch.from_numpy(fall_windows).to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()

    config = json.loads(paths["fall_decision_config.json"].read_text(encoding="utf-8"))
    rolling_count = int(config.get("rolling_window_count", 3))
    threshold = float(config.get("recording_threshold", 0.99))
    if len(probabilities) >= rolling_count:
        rolling = np.convolve(probabilities, np.ones(rolling_count) / rolling_count, mode="valid")
        best_index = int(np.argmax(rolling))
        score = float(rolling[best_index])
        risk_start = float(pose_times[best_index])
        risk_end_index = min(best_index + 5 + rolling_count - 1, len(pose_times) - 1)
        risk_end = float(pose_times[risk_end_index])
    else:
        rolling = probabilities.copy()
        best_index = int(np.argmax(rolling))
        score = float(rolling[best_index])
        risk_start = float(pose_times[best_index])
        risk_end = float(pose_times[min(best_index + 5, len(pose_times) - 1)])

    label = "낙상" if score >= threshold else "비낙상"
    output_dir = output_dir or csv_path.parent / f"{csv_path.stem}_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_pose = pose.reshape(-1, 33, 2) * np.exp(log_scale)[:, None, :] + center[:, None, :]
    pose_data = {"time_sec": pose_times}
    for joint in range(33):
        pose_data[f"kp_{joint}_x"] = raw_pose[:, joint, 0]
        pose_data[f"kp_{joint}_y"] = raw_pose[:, joint, 1]
    pd.DataFrame(pose_data).to_csv(output_dir / "predicted_pose.csv", index=False)

    fall_starts = pose_times[:len(probabilities)]
    fall_ends = pose_times[5:5 + len(probabilities)]
    pd.DataFrame({
        "start_time_sec": fall_starts,
        "end_time_sec": fall_ends,
        "fall_probability": probabilities,
    }).to_csv(output_dir / "fall_probabilities.csv", index=False)

    summary = {
        "input_file": str(csv_path.resolve()),
        "device": str(device),
        "complete_csi_cycles": int(len(signals)),
        "pose_outputs": int(len(pose)),
        "fall_windows": int(len(probabilities)),
        "recording_score": score,
        "threshold": threshold,
        "prediction": "fall" if label == "낙상" else "non_fall",
        "prediction_ko": label,
        "highest_risk_start_sec": risk_start,
        "highest_risk_end_sec": risk_end,
    }
    (output_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 54)
    print(f"최종 판정       : {label}")
    print(f"낙상 점수       : {score:.6f}")
    print(
        f"구간 확률       : 최소 {float(probabilities.min()):.6f} / "
        f"평균 {float(probabilities.mean()):.6f} / 최대 {float(probabilities.max()):.6f}"
    )
    print(f"판정 임계값     : {threshold:.6f}")
    print(f"최고 위험 구간  : {risk_start:.3f}초 ~ {risk_end:.3f}초")
    print(f"결과 저장 폴더  : {output_dir.resolve()}")
    print("=" * 54)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CareWave CSI 낙상/비낙상 통합 추론")
    parser.add_argument("csv", type=Path, help="원본 CSI CSV 경로")
    parser.add_argument("--model-dir", type=Path, default=None, help="모델 파일 5개가 있는 폴더")
    parser.add_argument("--output-dir", type=Path, default=None, help="결과 저장 폴더")
    parser.add_argument("--cpu", action="store_true", help="GPU가 있어도 CPU 사용")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"CSV 파일을 찾을 수 없습니다: {args.csv}")
    predict(args.csv, args.model_dir, args.output_dir, args.cpu)


if __name__ == "__main__":
    main()
