import pandas as pd
from pathlib import Path

DATASET_DIR = Path("carewave_dataset")
CSI_RAW_DIR = DATASET_DIR / "raw" / "csi"
CSI_OUT_DIR = DATASET_DIR / "processed" / "csi"

CSI_OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 10
TIME_BIN = 1 / TARGET_FPS
SKIP_EXISTING = True

SUBCARRIER_COLS = [f"sub_{i}" for i in range(52)]


def is_test_file(path: Path):
    return "test" in path.stem.lower() or path.parent.name.lower() == "test"


def read_csi_csv(csv_path: Path):
    fixed_columns = ["experiment_id", "timestamp", "label", "rx"] + SUBCARRIER_COLS

    try:
        df = pd.read_csv(csv_path)

        if "sub_51" not in df.columns:
            print(f"[WARN] sub_51 헤더 없음 → 컬럼명 보정: {csv_path.name}")
            df = pd.read_csv(
                csv_path,
                header=None,
                names=fixed_columns,
                skiprows=1,
                engine="python",
                on_bad_lines="skip",
            )

        return df

    except pd.errors.ParserError:
        print(f"[WARN] CSV 파싱 오류 → 컬럼명 보정: {csv_path.name}")
        df = pd.read_csv(
            csv_path,
            header=None,
            names=fixed_columns,
            skiprows=1,
            engine="python",
            on_bad_lines="skip",
        )
        return df


def get_rx_num(rx_id):
    rx_text = str(rx_id).lower().replace("rx", "").strip()
    return int(rx_text)


def preprocess_one_csi(csv_path: Path):
    sample_id = csv_path.stem.replace("_csi_raw", "")
    output_path = CSI_OUT_DIR / f"{sample_id}_csi_10fps.csv"

    if is_test_file(csv_path):
        print(f"[SKIP TEST] {csv_path.name}")
        return

    if SKIP_EXISTING and output_path.exists():
        print(f"[SKIP EXISTING] {output_path.name}")
        return

    df = read_csi_csv(csv_path)

    if "timestamp" not in df.columns:
        print(f"[SKIP] timestamp 없음: {csv_path.name}")
        return

    if "rx" not in df.columns:
        print(f"[SKIP] rx 없음: {csv_path.name}")
        return

    missing_subs = [c for c in SUBCARRIER_COLS if c not in df.columns]
    if missing_subs:
        print(f"[SKIP] subcarrier 컬럼 부족: {csv_path.name}")
        print(f"부족한 컬럼: {missing_subs}")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if df.empty:
        print(f"[SKIP] 유효 timestamp 없음: {csv_path.name}")
        return

    start_time = df["timestamp"].min()
    df["time_sec"] = (df["timestamp"] - start_time).dt.total_seconds()

    df["time_bin"] = (df["time_sec"] / TIME_BIN).round().astype(int)

    rows = []

    for time_bin, group in df.groupby("time_bin"):
        row = {
            "sample_id": sample_id,
            "time_sec": time_bin * TIME_BIN,
        }

        for rx_id in sorted(group["rx"].dropna().unique()):
            rx_group = group[group["rx"] == rx_id]
            rx_num = get_rx_num(rx_id)

            for sub in SUBCARRIER_COLS:
                row[f"rx{rx_num}_{sub}"] = rx_group[sub].mean()

        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("time_sec").reset_index(drop=True)

    numeric_cols = out_df.select_dtypes(include=["number"]).columns
    out_df[numeric_cols] = out_df[numeric_cols].interpolate(limit_direction="both")

    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[OK] {csv_path.name} → {output_path.name} / {len(out_df)} rows")


def main():
    csv_files = sorted(CSI_RAW_DIR.glob("*/*_csi_raw.csv"))

    print(f"전체 CSI raw 파일 개수: {len(csv_files)}")

    processed_count = 0

    for csv_path in csv_files:
        sample_id = csv_path.stem.replace("_csi_raw", "")
        output_path = CSI_OUT_DIR / f"{sample_id}_csi_10fps.csv"
        before_exists = output_path.exists()

        preprocess_one_csi(csv_path)

        after_exists = output_path.exists()
        if not before_exists and after_exists and not is_test_file(csv_path):
            processed_count += 1

    print("\nCSI 전처리 완료")
    print(f"새로 처리된 CSI 개수: {processed_count}")


if __name__ == "__main__":
    main()