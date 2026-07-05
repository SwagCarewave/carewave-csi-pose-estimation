from pathlib import Path

import numpy as np
import torch

from train_sequence_model import CSIPoseSequenceLSTM

DATASET_DIR = Path("carewave_dataset")
DATA_DIR = DATASET_DIR / "processed" / "dataset"
MODEL_PATH = DATASET_DIR / "models" / "csi_pose_lstm_sequence_best.pth"


def weighted_mse_np(pred, target, visibility):
    weights = np.clip(visibility, 0.05, 1.0)
    error = ((pred - target) ** 2) * weights
    return float(error.sum() / max(weights.sum(), 1.0))


def predict_split(model, x, indices, device, batch_size=512):
    preds = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = torch.tensor(x[batch_idx], dtype=torch.float32).to(device)
            pred = model(batch).cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds, axis=0) if preds else np.empty((0, 10, 66))


def evaluate_split(model, x, y, visibility, indices, device, batch_size=512):
    pred_array = predict_split(model, x, indices, device, batch_size=batch_size)
    target = y[indices]
    weights = visibility[indices]

    return weighted_mse_np(pred_array, target, weights)


def denormalize_pose(flat_pose, center, scale):
    pose = flat_pose.reshape(flat_pose.shape[0], flat_pose.shape[1], 33, 2)
    scale = np.maximum(scale, 1e-3)
    pose = pose * scale[:, :, None, :] + center[:, :, None, :]
    return pose.reshape(flat_pose.shape)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = CSIPoseSequenceLSTM(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        output_dim=checkpoint["output_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    target = checkpoint.get("target", "raw")
    x = np.load(DATA_DIR / "X_csi.npy").astype(np.float32)
    pose_norm = np.load(DATA_DIR / "Y_pose_seq_norm.npy").astype(np.float32).reshape(len(x), 10, 66)
    y_raw = np.load(DATA_DIR / "Y_pose_seq.npy").astype(np.float32).reshape(len(x), 10, 66)
    center = np.load(DATA_DIR / "Y_pose_center_seq.npy").astype(np.float32)
    scale = np.load(DATA_DIR / "Y_pose_scale_seq.npy").astype(np.float32)
    visibility = np.load(DATA_DIR / "Y_visibility_seq.npy").astype(np.float32)
    pose_visibility = np.repeat(visibility[..., None], 2, axis=-1).reshape(len(x), 10, 66)

    if target == "raw":
        y = y_raw
        target_visibility = pose_visibility
    elif target == "raw_and_normalized":
        y = np.concatenate([y_raw, pose_norm], axis=-1)
        target_visibility = np.concatenate([pose_visibility, pose_visibility], axis=-1)
    elif target == "normalized":
        y = pose_norm
        target_visibility = pose_visibility
    else:
        y = np.concatenate([pose_norm, center, scale], axis=-1)
        frame_visibility = np.ones((len(x), 10, 3), dtype=np.float32)
        target_visibility = np.concatenate([pose_visibility, frame_visibility], axis=-1)

    splits = np.load(DATA_DIR / "split_indices.npz")

    x = (x - checkpoint["x_mean"]) / checkpoint["x_std"]

    print("model:", MODEL_PATH)
    print("target:", target)
    print("best_epoch:", checkpoint.get("best_epoch"))
    print("best_val_loss:", checkpoint.get("best_val_loss"))

    for split_name in ["train", "val", "test"]:
        indices = splits[f"{split_name}_idx"]
        pred = predict_split(model, x, indices, device)
        loss = weighted_mse_np(pred, y[indices], target_visibility[indices])

        if target == "normalized":
            pred_raw = denormalize_pose(pred, center[indices], scale[indices])
        elif target == "normalized_with_frame":
            pred_raw = denormalize_pose(
                pred[..., :66],
                pred[..., 66:68],
                pred[..., 68:69],
            )
        elif target == "raw_and_normalized":
            pred_raw = pred[..., :66]
        else:
            pred_raw = pred

        raw_loss = weighted_mse_np(pred_raw, y_raw[indices], pose_visibility[indices])
        print(
            f"{split_name} weighted_mse: {loss:.6f} "
            f"raw_mse: {raw_loss:.6f} "
            f"({len(indices)} windows)"
        )


if __name__ == "__main__":
    main()
