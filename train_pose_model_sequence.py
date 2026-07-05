# train_pose_model_sequence.py

import os
import joblib
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATASET_DIR = "dataset_sequence_all"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

X = np.load(os.path.join(DATASET_DIR, "X_csi_seq.npy"))
Y = np.load(os.path.join(DATASET_DIR, "Y_pose_seq.npy"))

print("X shape:", X.shape)  # (samples, seq_len, features)
print("Y shape:", Y.shape)

num_samples, seq_len, num_features = X.shape

# scaler는 feature 축 기준으로 적용해야 하므로 2D로 펼친 뒤 다시 복원
X_2d = X.reshape(-1, num_features)

scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)

joblib.dump(scaler, os.path.join(MODEL_DIR, "csi_sequence_scaler.pkl"))

X_scaled = X_2d_scaled.reshape(num_samples, seq_len, num_features)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled,
    Y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

model = Sequential([
    Conv1D(64, kernel_size=3, padding="same", activation="relu",
           input_shape=(seq_len, num_features)),
    BatchNormalization(),

    LSTM(128, return_sequences=True),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    Dropout(0.3),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    BatchNormalization(),

    Dense(64, activation="relu"),

    Dense(Y.shape[1])
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.Huber(delta=0.05),
    metrics=["mae"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=25,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "pose_prediction_model_sequence_best.keras"),
    monitor="val_loss",
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=8,
    min_lr=1e-5
)

history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=300,
    batch_size=32,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

loss, mae = model.evaluate(X_test, Y_test)

print("\n===== 평가 결과 =====")
print("Test Loss:", loss)
print("Test MAE:", mae)

model.save(os.path.join(MODEL_DIR, "pose_prediction_model_sequence.h5"))

print("모델 저장 완료:", os.path.join(MODEL_DIR, "pose_prediction_model_sequence.h5"))
print("스케일러 저장 완료:", os.path.join(MODEL_DIR, "csi_sequence_scaler.pkl"))