#!/usr/bin/env python3
"""Compare exported PINN CSV files with HEC-RAS validation truth."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["time_id", "cell_id"]
FILE_SPECS = {
    "hydrodynamics": (
        "hydrodynamics_truth.csv",
        "pinn_hydrodynamics.csv",
        [
            "water_surface_m",
            "depth_m",
            "velocity_x_mps",
            "velocity_y_mps",
        ],
    ),
    "sediment": (
        "sediment_truth.csv",
        "pinn_sediment.csv",
        None,
    ),
    "bed": (
        "bed_truth.csv",
        "pinn_bed.csv",
        [
            "bed_elevation_m",
            "bed_change_m",
            "d10_mm",
            "d16_mm",
            "d50_mm",
            "d90_mm",
        ],
    ),
}


class MetricAccumulator:
    def __init__(self):
        self.count = 0
        self.sum_error = 0.0
        self.sum_abs_error = 0.0
        self.sum_squared_error = 0.0
        self.max_abs_error = 0.0
        self.sum_truth = 0.0
        self.sum_truth_squared = 0.0

    def update(self, truth, prediction):
        valid = np.isfinite(truth) & np.isfinite(prediction)
        truth = truth[valid].astype(np.float64)
        prediction = prediction[valid].astype(np.float64)
        if truth.size == 0:
            return
        error = prediction - truth
        self.count += truth.size
        self.sum_error += error.sum()
        self.sum_abs_error += np.abs(error).sum()
        self.sum_squared_error += np.square(error).sum()
        self.max_abs_error = max(self.max_abs_error, float(np.abs(error).max()))
        self.sum_truth += truth.sum()
        self.sum_truth_squared += np.square(truth).sum()

    def result(self):
        if self.count == 0:
            return {}
        mean_truth = self.sum_truth / self.count
        total_variance = (
            self.sum_truth_squared - self.count * mean_truth ** 2
        )
        r2 = (
            1.0 - self.sum_squared_error / total_variance
            if total_variance > 0.0 else np.nan
        )
        return {
            "count": self.count,
            "mae": self.sum_abs_error / self.count,
            "rmse": np.sqrt(self.sum_squared_error / self.count),
            "bias": self.sum_error / self.count,
            "max_abs_error": self.max_abs_error,
            "r2": r2,
            "truth_mean": mean_truth,
        }


def _latest_inference_dir(output_dir):
    directories = sorted(
        output_dir.glob("pinn_inference_*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not directories:
        raise FileNotFoundError(f"No pinn_inference_* directory in {output_dir}")
    return directories[-1]


def _comparison_columns(kind, truth_path, prediction_path, configured):
    truth_columns = pd.read_csv(truth_path, nrows=0).columns
    prediction_columns = pd.read_csv(prediction_path, nrows=0).columns
    common = [
        name for name in truth_columns
        if name in prediction_columns and name not in {
            "time_id", "timestamp", "time_seconds", "cell_id", "x_m", "y_m",
            "wet", "velocity_reconstruction_valid",
        }
    ]
    if configured is not None:
        common = [name for name in configured if name in common]
    if kind == "sediment":
        common = [
            name for name in common
            if name.startswith("concentration_")
            or name.startswith("capacity_")
            or name in {
                "total_concentration_kg_m3",
                "total_capacity_kg_m3",
                "bed_shear_total_pa",
                "bed_shear_skin_pa",
            }
        ]
    return common


def compare_file(kind, truth_path, prediction_path, output_dir, columns, chunk_size):
    accumulators = {column: MetricAccumulator() for column in columns}
    per_time = {}
    final_rows = None

    truth_reader = pd.read_csv(truth_path, chunksize=chunk_size)
    prediction_reader = pd.read_csv(prediction_path, chunksize=chunk_size)
    for truth, prediction in zip(truth_reader, prediction_reader):
        if not np.array_equal(truth[KEYS].to_numpy(), prediction[KEYS].to_numpy()):
            raise ValueError(f"{kind}: PINN and HEC-RAS row keys do not align.")

        if kind == "hydrodynamics":
            mask = truth["wet"].to_numpy(dtype=bool)
            velocity_mask = mask & truth[
                "velocity_reconstruction_valid"
            ].to_numpy(dtype=bool)
        else:
            mask = np.ones(len(truth), dtype=bool)
            velocity_mask = mask

        time_ids = truth["time_id"].to_numpy()
        for column in columns:
            column_mask = (
                velocity_mask
                if column in {"velocity_x_mps", "velocity_y_mps"}
                else mask
            )
            truth_values = truth[column].to_numpy()[column_mask]
            prediction_values = prediction[column].to_numpy()[column_mask]
            accumulators[column].update(truth_values, prediction_values)
            for time_id in np.unique(time_ids[column_mask]):
                local = column_mask & (time_ids == time_id)
                key = (int(time_id), column)
                per_time.setdefault(key, MetricAccumulator()).update(
                    truth[column].to_numpy()[local],
                    prediction[column].to_numpy()[local],
                )

        max_time = int(truth["time_id"].max())
        if final_rows is None or max_time >= int(final_rows["time_id"].iloc[0]):
            candidate_mask = truth["time_id"] == max_time
            truth_final = truth.loc[candidate_mask, KEYS + columns].copy()
            prediction_final = prediction.loc[candidate_mask, columns]
            for column in columns:
                truth_final[f"{column}_pinn"] = prediction_final[column].to_numpy()
                truth_final[f"{column}_difference"] = (
                    prediction_final[column].to_numpy()
                    - truth_final[column].to_numpy()
                )
            final_rows = truth_final

    summary_rows = []
    for column, accumulator in accumulators.items():
        row = {"group": kind, "field": column}
        row.update(accumulator.result())
        summary_rows.append(row)

    per_time_rows = []
    for (time_id, column), accumulator in sorted(per_time.items()):
        row = {"group": kind, "time_id": time_id, "field": column}
        row.update(accumulator.result())
        per_time_rows.append(row)

    if final_rows is not None:
        final_rows.to_csv(output_dir / f"{kind}_final_time_difference.csv", index=False)
    return summary_rows, per_time_rows


def compare(args):
    inference_dir = args.inference_dir or _latest_inference_dir(args.output_dir)
    comparison_dir = args.comparison_dir or inference_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    per_time_rows = []

    for kind, (truth_name, prediction_name, configured) in FILE_SPECS.items():
        truth_path = args.truth_dir / truth_name
        prediction_path = inference_dir / prediction_name
        columns = _comparison_columns(
            kind,
            truth_path,
            prediction_path,
            configured,
        )
        group_summary, group_per_time = compare_file(
            kind,
            truth_path,
            prediction_path,
            comparison_dir,
            columns,
            args.chunk_size,
        )
        summary_rows.extend(group_summary)
        per_time_rows.extend(group_per_time)
        print(f"Compared {kind}: {len(columns)} fields")

    pd.DataFrame(summary_rows).to_csv(
        comparison_dir / "summary_metrics.csv",
        index=False,
    )
    pd.DataFrame(per_time_rows).to_csv(
        comparison_dir / "per_time_metrics.csv",
        index=False,
    )
    metadata = {
        "truth_dir": str(args.truth_dir),
        "inference_dir": str(inference_dir),
        "metric_definition": {
            "error": "PINN - HEC-RAS",
            "mae": "mean absolute error",
            "rmse": "root mean squared error",
            "bias": "mean signed error",
            "r2": "coefficient of determination",
        },
        "hydrodynamic_mask": (
            "depth and water surface use HEC-RAS wet cells; velocity additionally "
            "requires velocity_reconstruction_valid=1"
        ),
    }
    (comparison_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Comparison saved to: {comparison_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs-autodl"))
    parser.add_argument("--inference-dir", type=Path)
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=Path("data/Chippewa_2D/validation_truth"),
    )
    parser.add_argument("--comparison-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=100000)
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
