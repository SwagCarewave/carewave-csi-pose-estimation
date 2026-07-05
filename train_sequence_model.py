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

MODEL_PATH = MODEL_DIR / "csi_pose_lstm_sequence_best.pth"


class CSIPoseSequenceLSTM(nn.Module):
    def __init__(self, input_dim=156, hidden_dim=256, num_layers=2, output_dim=66):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


def load_arrays(target):
    x = np.load(DATA_DIR / "X_csi.npy").astype(np.float32)
    visibility = np.load(DATA_DIR / "Y_visibility_seq.npy").astype(np.float32)
    split_indices = np.load(DATA_DIR / "split_indices.npz")

    if target == "raw":
        y = np.load(DATA_DIR / "Y_pose_seq.npy").astype(np.float32)
        y = y.reshape(y.shape[0], y.shape[1], -1)
        visibility = np.repeat(visibility[..., None], 2, axis=-1).reshape(y.shape)
    elif target == "normalized":
        y = np.load(DATA_DIR / "Y_pose_seq_norm.npy").astype(np.float32)
        y = y.reshape(y.shape[0], y.shape[1], -1)
        visibility = np.repeat(visibility[..., None], 2, axis=-1).reshape(y.shape)
    elif target == "raw_and_normalized":
        raw_pose = np.load(DATA_DIR / "Y_pose_seq.npy").astype(np.float32)
        norm_pose = np.load(DATA_DIR / "Y_pose_seq_norm.npy").astype(np.float32)

        raw_pose = raw_pose.reshape(raw_pose.shape[0], raw_pose.shape[1], -1)
        norm_pose = norm_pose.reshape(norm_pose.shape[0], norm_pose.shape[1], -1)
        y = np.concatenate([raw_pose, norm_pose], axis=-1)

        pose_visibility = np.repeat(visibility[..., None], 2, axis=-1).reshape(raw_pose.shape)
        visibility = np.concatenate([pose_visibility, pose_visibility], axis=-1)
    else:
        pose = np.load(DATA_DIR / "Y_pose_seq_norm.npy").astype(np.float32)
        center = np.load(DATA_DIR / "Y_pose_center_seq.npy").astype(np.float32)
        scale = np.load(DATA_DIR / "Y_pose_scale_seq.npy").astype(np.float32)

        pose = pose.reshape(pose.shape[0], pose.shape[1], -1)
        y = np.concatenate([pose, center, scale], axis=-1)

        pose_visibility = np.repeat(visibility[..., None], 2, axis=-1).reshape(pose.shape)
        frame_visibility = np.ones((*pose.shape[:2], 3), dtype=np.float32)
        visibility = np.concatenate([pose_visibility, frame_visibility], axis=-1)

    return x, y, visibility, split_indices


def make_loader(x, y, visibility, indices, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(x[indices], dtype=torch.float32),
        torch.tensor(y[indices], dtype=torch.float32),
        torch.tensor(visibility[indices], dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def weighted_mse(pred, target, visibility):
    weights = visibility.clamp(0.05, 1.0)
    squared_error = (pred - target).pow(2) * weights
    return squared_error.sum() / weights.sum().clamp_min(1.0)


def temporal_smoothness_loss(pred):
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)

    return (pred[:, 1:] - pred[:, :-1]).pow(2).mean()


def run_epoch(model, loader, optimizer, device, smoothness_weight):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_pose_loss = 0.0
    total_items = 0

    for x, y, visibility in loader:
        x = x.to(device)
        y = y.to(device)
        visibility = visibility.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            pose_loss = weighted_mse(pred, y, visibility)
            smooth_loss = temporal_smoothness_loss(pred)
            loss = pose_loss + smoothness_weight * smooth_loss

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_pose_loss += pose_loss.item() * batch_size
        total_items += batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "pose_loss": total_pose_loss / max(total_items, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--aux-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--target",
        choices=["raw_and_normalized", "normalized_with_frame", "normalized", "raw"],
        default="raw_and_normalized",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    x, y, visibility, split_indices = load_arrays(args.target)
    if args.target == "raw_and_normalized":
        visibility[..., 66:] *= args.aux_loss_weight
    train_idx = split_indices["train_idx"]
    val_idx = split_indices["val_idx"]
    test_idx = split_indices["test_idx"]

    x_mean = x[train_idx].mean(axis=(0, 1), keepdims=True).astype(np.float32)
    x_std = x[train_idx].std(axis=(0, 1), keepdims=True).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    x = (x - x_mean) / x_std

    train_loader = make_loader(x, y, visibility, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(x, y, visibility, val_idx, args.batch_size, shuffle=False)
    test_loader = make_loader(x, y, visibility, test_idx, args.batch_size, shuffle=False)

    model = CSIPoseSequenceLSTM(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=y.shape[2],
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
            args.smoothness_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            smoothness_weight=args.smoothness_weight,
        )
        scheduler.step(val_metrics["loss"])

        print(
            f"epoch {epoch:03d} "
            f"train={train_metrics['loss']:.6f} "
            f"val={val_metrics['loss']:.6f} "
            f"val_pose={val_metrics['pose_loss']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "window_size": x.shape[1],
                    "input_dim": x.shape[2],
                    "output_dim": y.shape[2],
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "target_kind": f"pose_sequence_{args.target}",
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "target": args.target,
                    "aux_loss_weight": args.aux_loss_weight,
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
        smoothness_weight=args.smoothness_weight,
    )
    print(f"best epoch: {best_epoch}")
    print(f"test loss: {test_metrics['loss']:.6f}")
    print(f"test pose loss: {test_metrics['pose_loss']:.6f}")


if __name__ == "__main__":
    main()
