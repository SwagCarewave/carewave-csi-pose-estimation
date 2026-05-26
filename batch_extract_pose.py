# 모든 영상에서 관절 추출하기

import cv2
import mediapipe as mp
import pandas as pd
import os
import glob

RAW_DIR = "dataset_raw"
OUTPUT_DIR = "output_pose"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose

def extract_pose(video_path, output_csv_path, skip_frames=2):
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] 영상 열기 실패:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n[START]", video_path)
    print("FPS:", fps)
    print("총 프레임:", total_frames)

    data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        row = {
            "frame": frame_idx,
            "time_sec": frame_idx / fps
        }

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                row[f"x{i}"] = lm.x
                row[f"y{i}"] = lm.y
                row[f"z{i}"] = lm.z
                row[f"visibility{i}"] = lm.visibility
        else:
            for i in range(33):
                row[f"x{i}"] = None
                row[f"y{i}"] = None
                row[f"z{i}"] = None
                row[f"visibility{i}"] = None

        data.append(row)

        if frame_idx % 600 == 0:
            print(f"{frame_idx}/{total_frames} 처리 중...")

        frame_idx += 1

    cap.release()
    pose.close()

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print("[DONE] 저장:", output_csv_path)
    print("pose rows:", len(df))


video_files = glob.glob(os.path.join(RAW_DIR, "**", "*.mp4"), recursive=True)

print("찾은 영상 개수:", len(video_files))

for video_path in video_files:
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv_path = os.path.join(OUTPUT_DIR, f"{base_name}_pose.csv")

    if os.path.exists(output_csv_path):
        print("[SKIP] 이미 존재:", output_csv_path)
        continue

    extract_pose(video_path, output_csv_path, skip_frames=2)