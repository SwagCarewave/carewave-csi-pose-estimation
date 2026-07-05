import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

DATASET_DIR = Path("carewave_dataset")
VIDEO_DIR = DATASET_DIR / "raw" / "videos"
POSE_DIR = DATASET_DIR / "processed" / "pose"

POSE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 10
SKIP_EXISTING = True

mp_pose = mp.solutions.pose


def is_test_file(path: Path):
    return "test" in path.stem.lower() or path.parent.name.lower() == "test"


def extract_pose_from_video(video_path: Path):
    sample_id = video_path.stem
    output_csv = POSE_DIR / f"{sample_id}_pose.csv"

    # 학습용 전처리에서는 test 영상 제외
    if is_test_file(video_path):
        print(f"[SKIP TEST] {video_path.name}")
        return

    # 이미 pose csv가 있으면 다시 처리하지 않음
    if SKIP_EXISTING and output_csv.exists():
        print(f"[SKIP EXISTING] {output_csv.name}")
        return

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        print(f"[ERROR] FPS 읽기 실패: {video_path}")
        cap.release()
        return

    frame_interval = max(1, round(original_fps / TARGET_FPS))

    rows = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            time_sec = frame_idx / original_fps

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)

            row = {
                "sample_id": sample_id,
                "frame_idx": frame_idx,
                "pose_idx": saved_idx,
                "time_sec": time_sec,
                "original_fps": original_fps,
                "target_fps": TARGET_FPS,
            }

            if result.pose_landmarks:
                for i, lm in enumerate(result.pose_landmarks.landmark):
                    row[f"kp_{i}_x"] = lm.x
                    row[f"kp_{i}_y"] = lm.y
                    row[f"kp_{i}_z"] = lm.z
                    row[f"kp_{i}_visibility"] = lm.visibility
            else:
                for i in range(33):
                    row[f"kp_{i}_x"] = None
                    row[f"kp_{i}_y"] = None
                    row[f"kp_{i}_z"] = None
                    row[f"kp_{i}_visibility"] = None

            rows.append(row)

            saved_idx += 1
            frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] {video_path.name} → {output_csv.name} / {len(df)} rows")


def main():
    video_files = sorted(VIDEO_DIR.glob("*/*.mp4"))

    print(f"전체 영상 개수: {len(video_files)}")

    processed_count = 0

    for video_path in video_files:
        before_exists = (POSE_DIR / f"{video_path.stem}_pose.csv").exists()
        extract_pose_from_video(video_path)

        after_exists = (POSE_DIR / f"{video_path.stem}_pose.csv").exists()
        if not before_exists and after_exists and not is_test_file(video_path):
            processed_count += 1

    print("\nPose 추출 완료")
    print(f"새로 처리된 영상 개수: {processed_count}")


if __name__ == "__main__":
    main()