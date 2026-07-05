import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from visualize_pose import draw_pose

DATASET_DIR = Path("carewave_dataset")
MODEL_PATH = DATASET_DIR / "models" / "csi_pose_lstm_best.pth"
X_PATH = DATASET_DIR / "processed" / "dataset" / "X_csi.npy"


class CSIPoseLSTM(nn.Module):
    def __init__(self, input_dim=156, hidden_dim=256, num_layers=2, output_dim=66):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = CSIPoseLSTM().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]

    X = np.load(X_PATH)

    sample = X[0:1]

    sample_norm = (sample - x_mean) / x_std
    sample_tensor = torch.tensor(sample_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(sample_tensor).cpu().numpy()

    pose = pred.reshape(33, 2)

    print("예측 pose shape:", pose.shape)
    print("nose:", pose[0])
    print("left_shoulder:", pose[11])
    print("right_shoulder:", pose[12])

    draw_pose(pose)


if __name__ == "__main__":
    main()