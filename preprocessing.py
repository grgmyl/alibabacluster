

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

COLS = ["id", "timestamp", "machine_id", "cpu_usage", "memory_usage"]

def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=COLS, low_memory=False)

    # Remove accidental header rows embedded as data
    df = df[df["id"] != "id"].copy()

    # Type conversion
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["cpu_usage"] = pd.to_numeric(df["cpu_usage"], errors="coerce")
    df["memory_usage"] = pd.to_numeric(df["memory_usage"], errors="coerce")
    df["machine_id"] = df["machine_id"].astype(str)

    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=["timestamp", "cpu_usage", "memory_usage"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["cpu_usage"] = df["cpu_usage"].interpolate(method="linear", limit_direction="both")
    df["memory_usage"] = df["memory_usage"].interpolate(method="linear", limit_direction="both")

    df["cpu_usage"] = df["cpu_usage"].bfill().ffill()
    df["memory_usage"] = df["memory_usage"].bfill().ffill()

    return df

def remove_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()

    for col in cols:
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out = out[(out[col] >= lower) & (out[col] <= upper)]

    return out.reset_index(drop=True)

def aggregate_per_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("timestamp")[["cpu_usage", "memory_usage"]]
        .mean()
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return agg

def infer_timestep_step(timestamps: pd.Series) -> int:
    diffs = timestamps.diff().dropna()
    if len(diffs) == 0:
        raise ValueError("Cannot infer timestep step: not enough timestamps.")

    step = int(diffs.mode().iloc[0])

    if step <= 0:
        raise ValueError(f"Invalid inferred timestep step: {step}")

    return step

def regularize_time_grid(agg: pd.DataFrame, step: int | None = None) -> pd.DataFrame:
    agg = agg.sort_values("timestamp").reset_index(drop=True)

    if step is None:
        step = infer_timestep_step(agg["timestamp"])

    full_index = np.arange(agg["timestamp"].min(), agg["timestamp"].max() + step, step)

    regular = (
        agg.set_index("timestamp")
        .reindex(full_index)
        .rename_axis("timestamp")
        .reset_index()
    )

    regular["cpu_usage"] = regular["cpu_usage"].interpolate(method="linear", limit_direction="both")
    regular["memory_usage"] = regular["memory_usage"].interpolate(method="linear", limit_direction="both")

    regular["cpu_usage"] = regular["cpu_usage"].bfill().ffill()
    regular["memory_usage"] = regular["memory_usage"].bfill().ffill()

    return regular

def build_lstm_ready_dataset(
    input_path: Path,
    remove_outliers: bool = True,
    regularize: bool = True,
    forced_step: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    df = load_raw_csv(input_path)
    raw_rows = len(df)

    df = basic_cleaning(df)
    cleaned_rows = len(df)

    unique_machines = df["machine_id"].nunique()

    if remove_outliers:
        df = remove_outliers_iqr(df, ["cpu_usage", "memory_usage"])

    post_outlier_rows = len(df)

    agg = aggregate_per_timestamp(df)
    aggregated_timesteps = len(agg)

    inferred_step = None
    if regularize:
        inferred_step = forced_step if forced_step is not None else infer_timestep_step(agg["timestamp"])
        agg = regularize_time_grid(agg, step=inferred_step)

    final_timesteps = len(agg)

    metadata = {
        "source_file": str(input_path.name),
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "post_outlier_rows": post_outlier_rows,
        "unique_machines": int(unique_machines),
        "aggregated_timesteps_before_regularization": aggregated_timesteps,
        "regularized": regularize,
        "inferred_step": inferred_step,
        "final_timesteps": final_timesteps,
    }

    return agg, metadata

def save_outputs(df_ready: pd.DataFrame, metadata: dict, output_csv: Path, output_meta: Path) -> None:
    df_ready.to_csv(output_csv, index=False)
    pd.DataFrame([metadata]).to_csv(output_meta, index=False)

    print("Saved preprocessed dataset to:", output_csv)
    print("Saved metadata to:", output_meta)
    print("\nSummary:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Node_0.csv for LSTM input.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw CSV file, e.g. /content/drive/MyDrive/project_5G/Node_0.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lstm_ready_aggregated.csv",
        help="Output CSV path for the final preprocessed series.",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="lstm_ready_metadata.csv",
        help="Output CSV path for preprocessing metadata.",
    )
    parser.add_argument(
        "--keep-outliers",
        action="store_true",
        help="Keep outliers instead of removing them with IQR filtering.",
    )
    parser.add_argument(
        "--no-regularize",
        action="store_true",
        help="Skip regularization to a fixed timestep grid.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Force a timestep step instead of inferring it from the data.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_csv = Path(args.output)
    output_meta = Path(args.meta)

    df_ready, metadata = build_lstm_ready_dataset(
        input_path=input_path,
        remove_outliers=not args.keep_outliers,
        regularize=not args.no_regularize,
        forced_step=args.step,
    )

    save_outputs(df_ready, metadata, output_csv, output_meta)

if __name__ == "__main__":
    main()

