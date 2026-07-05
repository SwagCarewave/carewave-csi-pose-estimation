from pathlib import Path

import numpy as np
import torch

from train_dual_head_sequence_model import CSIPoseDualHeadLSTM

DATASET_DIR = Path("carewave_dataset")
DATA_DIR = DATASET_DIR / "processed" / "dataset"
MODEL_PATH = DATASET_DIR / "models" / "csi_pose_lstm_dual_head_best.pth"


def weighted_mse_np(pred, target, visibility):
    weights = np.clip(visibility, 0.05, 1.0)
    error = ((pred - target) ** 2) * weights
    return float(error.sum() / max(weights.sum(), 1.0))


def predict_raw(model, x, indices, device, batch_size=512):
    preds = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = torch.tensor(x[batch_idx], dtype=torch.float32).to(device)
            pred = model(batch)["raw"].cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds, axis=0) if preds else np.empty((0, 10, 66))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = CSIPoseDualHeadLSTM(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    x = np.load(DATA_DIR / "X_csi.npy").astype(np.float32)
    y = np.load(DATA_DIR / "Y_pose_seq.npy").astype(np.float32).reshape(len(x), 10, 66)
    visibility = np.load(DATA_DIR / "Y_visibility_seq.npy").astype(np.float32)
    visibility = np.repeat(visibility[..., None], 2, axis=-1).reshape(len(x), 10, 66)
    splits = np.load(DATA_DIR / "split_indices.npz")

    x = (x - checkpoint["x_mean"]) / checkpoint["x_std"]

    print("model:", MODEL_PATH)
    print("target:", checkpoint.get("target"))
    print("best_epoch:", checkpoint.get("best_epoch"))
    print("best_val_raw_loss:", checkpoint.get("best_val_raw_loss"))

    for split_name in ["train", "val", "test"]:
        indices = splits[f"{split_name}_idx"]
        pred = predict_raw(model, x, indices, device)
        loss = weighted_mse_np(pred, y[indices], visibility[indices])
        print(f"{split_name} raw_mse: {loss:.6f} ({len(indices)} windows)")


if __name__ == "__main__":
    main()
