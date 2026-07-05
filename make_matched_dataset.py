import pandas as pd
from pathlib import Path

DATASET_DIR = Path("carewave_dataset")

POSE_DIR = DATASET_DIR / "processed" / "pose"
CSI_DIR = DATASET_DIR / "processed" / "csi"
MATCHED_DIR = DATASET_DIR / "processed" / "matched"

MATCHED_DIR.mkdir(parents=True, exist_ok=True)

TIME_TOLERANCE = 0.05
SKIP_EXISTING = True


def is_test_sample(sample_id: str):
    return "test" in sample_id.lower()


def get_sample_id_from_pose(path: Path):
    return path.stem.replace("_pose", "")


def get_sample_id_from_csi(path: Path):
    return path.stem.replace("_csi_10fps", "")


def match_one_sample(sample_id: str, pose_path: Path, csi_path: Path):
    output_path = MATCHED_DIR / f"{sample_id}_matched.csv"

    if is_test_sample(sample_id):
        print(f"[SKIP TEST] {sample_id}")
        return

    if SKIP_EXISTING and output_path.exists():
        print(f"[SKIP EXISTING] {output_path.name}")
        return

    pose_df = pd.read_csv(pose_path)
    csi_df = pd.read_csv(csi_path)

    if "time_sec" not in pose_df.columns:
        print(f"[SKIP] pose time_sec 없음: {pose_path.name}")
        return

    if "time_sec" not in csi_df.columns:
        print(f"[SKIP] csi time_sec 없음: {csi_path.name}")
        return

    pose_df = pose_df.sort_values("time_sec").reset_index(drop=True)
    csi_df = csi_df.sort_values("time_sec").reset_index(drop=True)

    matched_df = pd.merge_asof(
        pose_df,
        csi_df,
        on="time_sec",
        direction="nearest",
        tolerance=TIME_TOLERANCE,
        suffixes=("_pose", "_csi"),
    )

    before = len(matched_df)

    csi_cols = [c for c in matched_df.columns if c.startswith("rx")]
    matched_df = matched_df.dropna(subset=csi_cols)

    after = len(matched_df)

    matched_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[OK] {sample_id}: {before} rows → {after} matched rows")


def main():
    pose_files = sorted(
        p for p in POSE_DIR.glob("*_pose.csv")
        if "test" not in p.stem.lower()
    )

    csi_files = sorted(
        p for p in CSI_DIR.glob("*_csi_10fps.csv")
        if "test" not in p.stem.lower()
    )

    pose_map = {get_sample_id_from_pose(p): p for p in pose_files}
    csi_map = {get_sample_id_from_csi(c): c for c in csi_files}

    common_ids = sorted(set(pose_map.keys()) & set(csi_map.keys()))

    pose_only = sorted(set(pose_map.keys()) - set(csi_map.keys()))
    csi_only = sorted(set(csi_map.keys()) - set(pose_map.keys()))

    print(f"pose 파일 개수: {len(pose_files)}")
    print(f"csi 파일 개수: {len(csi_files)}")
    print(f"매칭 가능한 sample 개수: {len(common_ids)}")

    if pose_only:
        print("\n[pose만 있고 csi 없는 sample]")
        for x in pose_only:
            print("-", x)

    if csi_only:
        print("\n[csi만 있고 pose 없는 sample]")
        for x in csi_only:
            print("-", x)

    print("\n매칭 시작\n")

    processed_count = 0

    for sample_id in common_ids:
        output_path = MATCHED_DIR / f"{sample_id}_matched.csv"
        before_exists = output_path.exists()

        match_one_sample(sample_id, pose_map[sample_id], csi_map[sample_id])

        after_exists = output_path.exists()
        if not before_exists and after_exists and not is_test_sample(sample_id):
            processed_count += 1

    print("\n매칭 완료")
    print(f"새로 생성된 matched 개수: {processed_count}")


if __name__ == "__main__":
    main()