import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATASET_DIR = Path("carewave_dataset")
DATA_DIR = DATASET_DIR / "processed" / "dataset"
MODEL_DIR = DATASET_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "csi_pose_lstm_dual_head_best.pth"


class CSIPoseDualHeadLSTM(nn.Module):
    def __init__(self, input_dim=156, hidden_dim=256, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.raw_head = nn.Linear(256, 66)
        self.norm_head = nn.Linear(256, 66)

    def forward(self, x):
        out, _ = self.lstm(x)
        features = self.shared(out)
        return {
            "raw": self.raw_head(features),
            "normalized": self.norm_head(features),
        }


def weighted_mse(pred, target, visibility):
    weights = visibility.clamp(0.05, 1.0)
    error = (pred - target).pow(2) * weights
    return error.sum() / weights.sum().clamp_min(1.0)


def temporal_smoothness_loss(pred):
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)

    return (pred[:, 1:] - pred[:, :-1]).pow(2).mean()


def load_arrays():
    x = np.load(DATA_DIR / "X_csi.npy").astype(np.float32)
    raw = np.load(DATA_DIR / "Y_pose_seq.npy").astype(np.float32)
    normalized = np.load(DATA_DIR / "Y_pose_seq_norm.npy").astype(np.float32)
    visibility = np.load(DATA_DIR / "Y_visibility_seq.npy").astype(np.float32)
    splits = np.load(DATA_DIR / "split_indices.npz")

    raw = raw.reshape(raw.shape[0], raw.shape[1], -1)
    normalized = normalized.reshape(normalized.shape[0], normalized.shape[1], -1)
    visibility = np.repeat(visibility[..., None], 2, axis=-1).reshape(raw.shape)

    return x, raw, normalized, visibility, splits


def make_loader(x, raw, normalized, visibility, indices, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(x[indices], dtype=torch.float32),
        torch.tensor(raw[indices], dtype=torch.float32),
        torch.tensor(normalized[indices], dtype=torch.float32),
        torch.tensor(visibility[indices], dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer, device, aux_loss_weight, smoothness_weight):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_raw_loss = 0.0
    total_norm_loss = 0.0
    total_items = 0

    for x, raw, normalized, visibility in loader:
        x = x.to(device)
        raw = raw.to(device)
        normalized = normalized.to(device)
        visibility = visibility.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            raw_loss = weighted_mse(pred["raw"], raw, visibility)
            norm_loss = weighted_mse(pred["normalized"], normalized, visibility)
            smooth_loss = temporal_smoothness_loss(pred["raw"])
            loss = raw_loss + aux_loss_weight * norm_loss + smoothness_weight * smooth_loss

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_raw_loss += raw_loss.item() * batch_size
        total_norm_loss += norm_loss.item() * batch_size
        total_items += batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "raw_loss": total_raw_loss / max(total_items, 1),
        "norm_loss": total_norm_loss / max(total_items, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--aux-loss-weight", type=float, default=0.05)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    x, raw, normalized, visibility, splits = load_arrays()
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    x_mean = x[train_idx].mean(axis=(0, 1), keepdims=True).astype(np.float32)
    x_std = x[train_idx].std(axis=(0, 1), keepdims=True).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    x = (x - x_mean) / x_std

    train_loader = make_loader(
        x, raw, normalized, visibility, train_idx, args.batch_size, shuffle=True
    )
    val_loader = make_loader(
        x, raw, normalized, visibility, val_idx, args.batch_size, shuffle=False
    )
    test_loader = make_loader(
        x, raw, normalized, visibility, test_idx, args.batch_size, shuffle=False
    )

    model = CSIPoseDualHeadLSTM(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.aux_loss_weight,
            args.smoothness_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            aux_loss_weight=args.aux_loss_weight,
            smoothness_weight=args.smoothness_weight,
        )
        scheduler.step(val_metrics["raw_loss"])

        print(
            f"epoch {epoch:03d} "
            f"train_raw={train_metrics['raw_loss']:.6f} "
            f"val_raw={val_metrics['raw_loss']:.6f} "
            f"val_norm={val_metrics['norm_loss']:.6f}"
        )

        if val_metrics["raw_loss"] < best_val:
            best_val = val_metrics["raw_loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "window_size": x.shape[1],
                    "input_dim": x.shape[2],
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "target": "dual_head_raw_with_normalized_aux",
                    "best_val_raw_loss": best_val,
                    "best_epoch": best_epoch,
                    "aux_loss_weight": args.aux_loss_weight,
                    "smoothness_weight": args.smoothness_weight,
                },
                MODEL_PATH,
            )
            print("saved:", MODEL_PATH)

        if epoch - best_epoch >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        aux_loss_weight=args.aux_loss_weight,
        smoothness_weight=args.smoothness_weight,
    )
    print(f"best epoch: {best_epoch}")
    print(f"test raw loss: {test_metrics['raw_loss']:.6f}")
    print(f"test normalized aux loss: {test_metrics['norm_loss']:.6f}")


if __name__ == "__main__":
    main()
