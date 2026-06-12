#!/usr/bin/env python3
"""Extract HEC-RAS 2D results into validation datasets for the PINN model."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


FT_TO_M = 0.3048
CFS_TO_M3S = 0.028316846592
MG_L_TO_KG_M3 = 1.0e-3
PSF_TO_PA = 47.88025898033584

AREA = "Perimeter 1"
GEOMETRY = f"Geometry/2D Flow Areas/{AREA}"
BASE_TS = (
    "Results/Unsteady/Output/Output Blocks/Base Output/"
    f"Unsteady Time Series/2D Flow Areas/{AREA}"
)
SEDIMENT_TS = (
    "Results/Unsteady/Output/Output Blocks/Sediment Transport/"
    f"Unsteady Time Series/2D Flow Areas/{AREA}"
)
BED_TS = f"Bed Time Series/2D Flow Areas/{AREA}"


def _decode(values):
    return np.asarray([
        value.decode("utf-8", errors="replace").strip()
        if isinstance(value, bytes) else str(value).strip()
        for value in values
    ])


def _write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _write_cell_truth_csvs(
    output_dir,
    time_stamps,
    time_seconds,
    coordinates_m,
    water_surface_m,
    depth_m,
    wet_mask,
    velocity_reconstruction_valid,
    velocity_x_mps,
    velocity_y_mps,
    available_names,
    concentration_kg_m3,
    capacity_kg_m3,
    total_concentration_kg_m3,
    total_capacity_kg_m3,
    bed_shear_total_pa,
    bed_shear_skin_pa,
    bed_elevation_m,
    bed_change_m,
    d10_mm,
    d16_mm,
    d50_mm,
    d90_mm,
):
    n_time, n_cells = depth_m.shape
    common_header = ["time_id", "timestamp", "time_seconds", "cell_id", "x_m", "y_m"]

    with (output_dir / "hydrodynamics_truth.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(common_header + [
            "water_surface_m", "depth_m", "wet",
            "velocity_reconstruction_valid", "velocity_x_mps", "velocity_y_mps",
        ])
        for time_id in range(n_time):
            for cell_id in range(n_cells):
                writer.writerow([
                    time_id,
                    time_stamps[time_id],
                    f"{time_seconds[time_id]:.6f}",
                    cell_id,
                    f"{coordinates_m[cell_id, 0]:.6f}",
                    f"{coordinates_m[cell_id, 1]:.6f}",
                    f"{water_surface_m[time_id, cell_id]:.7g}",
                    f"{depth_m[time_id, cell_id]:.7g}",
                    int(wet_mask[time_id, cell_id]),
                    int(velocity_reconstruction_valid[cell_id]),
                    f"{velocity_x_mps[time_id, cell_id]:.7g}",
                    f"{velocity_y_mps[time_id, cell_id]:.7g}",
                ])

    sediment_header = common_header + [
        f"concentration_{name}_kg_m3" for name in available_names
    ] + [
        "total_concentration_kg_m3",
    ] + [
        f"capacity_{name}_kg_m3" for name in available_names
    ] + [
        "total_capacity_kg_m3",
        "bed_shear_total_pa",
        "bed_shear_skin_pa",
    ]
    with (output_dir / "sediment_truth.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(sediment_header)
        for time_id in range(n_time):
            for cell_id in range(n_cells):
                writer.writerow([
                    time_id,
                    time_stamps[time_id],
                    f"{time_seconds[time_id]:.6f}",
                    cell_id,
                    f"{coordinates_m[cell_id, 0]:.6f}",
                    f"{coordinates_m[cell_id, 1]:.6f}",
                    *[
                        f"{value:.7g}"
                        for value in concentration_kg_m3[time_id, cell_id]
                    ],
                    f"{total_concentration_kg_m3[time_id, cell_id]:.7g}",
                    *[
                        f"{value:.7g}"
                        for value in capacity_kg_m3[time_id, cell_id]
                    ],
                    f"{total_capacity_kg_m3[time_id, cell_id]:.7g}",
                    f"{bed_shear_total_pa[time_id, cell_id]:.7g}",
                    f"{bed_shear_skin_pa[time_id, cell_id]:.7g}",
                ])

    with (output_dir / "bed_truth.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(common_header + [
            "bed_elevation_m", "bed_change_m",
            "d10_mm", "d16_mm", "d50_mm", "d90_mm",
        ])
        for time_id in range(n_time):
            for cell_id in range(n_cells):
                writer.writerow([
                    time_id,
                    time_stamps[time_id],
                    f"{time_seconds[time_id]:.6f}",
                    cell_id,
                    f"{coordinates_m[cell_id, 0]:.6f}",
                    f"{coordinates_m[cell_id, 1]:.6f}",
                    f"{bed_elevation_m[time_id, cell_id]:.7g}",
                    f"{bed_change_m[time_id, cell_id]:.7g}",
                    f"{d10_mm[time_id, cell_id]:.7g}",
                    f"{d16_mm[time_id, cell_id]:.7g}",
                    f"{d50_mm[time_id, cell_id]:.7g}",
                    f"{d90_mm[time_id, cell_id]:.7g}",
                ])


def _reconstruct_cell_velocity(hdf, face_velocity_mps):
    """Reconstruct cell-centered x/y velocity from signed face-normal velocity."""
    info = hdf[f"{GEOMETRY}/Cells Face and Orientation Info"][:]
    values = hdf[f"{GEOMETRY}/Cells Face and Orientation Values"][:]
    normals_and_length = hdf[f"{GEOMETRY}/Faces NormalUnitVector and Length"][:]
    normals = normals_and_length[:, :2].astype(np.float64)

    n_time = face_velocity_mps.shape[0]
    n_cells = info.shape[0]
    velocity_x = np.full((n_time, n_cells), np.nan, dtype=np.float32)
    velocity_y = np.full((n_time, n_cells), np.nan, dtype=np.float32)
    valid_cells = np.zeros(n_cells, dtype=bool)

    for cell_id, (start, count) in enumerate(info):
        face_ids = values[start:start + count, 0].astype(np.int64)
        if face_ids.size < 2:
            continue
        matrix = normals[face_ids]
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if (
            singular_values.size < 2
            or singular_values[1] <= singular_values[0] * 1.0e-4
        ):
            continue
        reconstruction = np.linalg.pinv(matrix, rcond=1.0e-4)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            cell_velocity = (
                face_velocity_mps[:, face_ids].astype(np.float64)
                @ reconstruction.T
            )
        if not np.isfinite(cell_velocity).all():
            continue
        velocity_x[:, cell_id] = cell_velocity[:, 0]
        velocity_y[:, cell_id] = cell_velocity[:, 1]
        valid_cells[cell_id] = True

    return velocity_x, velocity_y, valid_cells


def extract(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as hdf:
        coordinates_ft = hdf[f"{GEOMETRY}/Cells Center Coordinate"][:]
        coordinates_m = coordinates_ft * FT_TO_M
        surface_area_m2 = hdf[f"{GEOMETRY}/Cells Surface Area"][:] * FT_TO_M ** 2
        initial_manning = hdf[f"{GEOMETRY}/Cells Center Manning's n"][:]

        time_days = hdf["Bed Time Series/Time"][:]
        time_seconds = (time_days - time_days[0]) * 86400.0
        time_stamps = _decode(hdf["Bed Time Series/Time Date Stamp"][:])

        bed_elevation_m = hdf[f"{BED_TS}/Cell Bed Elevation"][:] * FT_TO_M
        bed_change_m = hdf[f"{BED_TS}/Cell Bed Change"][:] * FT_TO_M
        initial_bed_m = hdf[f"{BED_TS}/Cell Initial Bed Elevation"][0] * FT_TO_M
        d10_mm = hdf[f"{BED_TS}/Cell Active Layer Percentile Diameters - D10"][:]
        d16_mm = hdf[f"{BED_TS}/Cell Active Layer Percentile Diameters - D16"][:]
        d50_mm = hdf[f"{BED_TS}/Cell Active Layer Percentile Diameters - D50"][:]
        d90_mm = hdf[f"{BED_TS}/Cell Active Layer Percentile Diameters - D90"][:]

        water_surface_m = hdf[f"{BASE_TS}/Water Surface"][:] * FT_TO_M
        depth_m = np.maximum(water_surface_m - bed_elevation_m, 0.0).astype(np.float32)
        wet_mask = depth_m > 1.0e-5
        face_velocity_mps = hdf[f"{BASE_TS}/Face Velocity"][:] * FT_TO_M
        face_flow_m3s = hdf[f"{BASE_TS}/Face Flow"][:] * CFS_TO_M3S
        velocity_x_mps, velocity_y_mps, velocity_reconstruction_valid = (
            _reconstruct_cell_velocity(
            hdf,
            face_velocity_mps,
        )
        )
        velocity_x_mps[~wet_mask] = 0.0
        velocity_y_mps[~wet_mask] = 0.0
        velocity_x_mps[:, ~velocity_reconstruction_valid] = 0.0
        velocity_y_mps[:, ~velocity_reconstruction_valid] = 0.0

        grain_names_all = _decode(
            hdf["Sediment/Grain Class Data/Grain Class Names"][:]
        )
        grain_bounds_all = hdf["Sediment/Grain Class Data/Grain Class Bounds"][:]
        available_names = []
        for grain_name in grain_names_all:
            path = f"{SEDIMENT_TS}/Cell Total-load Concentration - {grain_name}"
            if path in hdf:
                available_names.append(grain_name)
        grain_indices = np.asarray([
            int(np.where(grain_names_all == name)[0][0])
            for name in available_names
        ])
        grain_bounds_mm = grain_bounds_all[grain_indices]

        concentration_mg_l = np.stack([
            hdf[f"{SEDIMENT_TS}/Cell Total-load Concentration - {name}"][:]
            for name in available_names
        ], axis=2)
        capacity_mg_l = np.stack([
            hdf[f"{SEDIMENT_TS}/Cell Total-load Capacity - {name}"][:]
            for name in available_names
        ], axis=2)
        total_concentration_mg_l = hdf[
            f"{SEDIMENT_TS}/Cell Total-load Concentration - Total"
        ][:]
        total_capacity_mg_l = hdf[
            f"{SEDIMENT_TS}/Cell Total-load Capacity - Total"
        ][:]
        bed_shear_total_pa = (
            hdf[f"{SEDIMENT_TS}/Cell Bed Shear Stress - Total"][:] * PSF_TO_PA
        )
        bed_shear_skin_pa = (
            hdf[f"{SEDIMENT_TS}/Cell Bed Shear Stress - Skin"][:] * PSF_TO_PA
        )

        upstream = hdf[
            "Event Conditions/Unsteady/Boundary Conditions/Flow Hydrographs/"
            f"2D: {AREA} BCLine: Upstream"
        ][:]
        downstream = hdf[
            "Event Conditions/Unsteady/Boundary Conditions/Stage Hydrographs/"
            f"2D: {AREA} BCLine: Downstream"
        ][:]

    _write_csv(
        output_dir / "cell_geometry.csv",
        [
            "cell_id", "x_ft", "y_ft", "x_m", "y_m",
            "surface_area_m2", "initial_bed_m", "initial_manning_n",
        ],
        (
            (
                cell_id,
                coordinates_ft[cell_id, 0],
                coordinates_ft[cell_id, 1],
                coordinates_m[cell_id, 0],
                coordinates_m[cell_id, 1],
                surface_area_m2[cell_id],
                initial_bed_m[cell_id],
                initial_manning[cell_id],
            )
            for cell_id in range(coordinates_m.shape[0])
        ),
    )
    _write_csv(
        output_dir / "time.csv",
        ["time_id", "timestamp", "time_days_raw", "time_seconds"],
        (
            (time_id, time_stamps[time_id], time_days[time_id], time_seconds[time_id])
            for time_id in range(time_days.size)
        ),
    )
    _write_csv(
        output_dir / "boundary_conditions.csv",
        ["boundary_id", "upstream_flow_cfs", "upstream_flow_m3s",
         "downstream_stage_ft", "downstream_stage_m"],
        (
            (
                index,
                upstream[index, 1],
                upstream[index, 1] * CFS_TO_M3S,
                downstream[index, 1],
                downstream[index, 1] * FT_TO_M,
            )
            for index in range(upstream.shape[0])
        ),
    )

    concentration_kg_m3 = (concentration_mg_l * MG_L_TO_KG_M3).astype(np.float32)
    capacity_kg_m3 = (capacity_mg_l * MG_L_TO_KG_M3).astype(np.float32)
    total_concentration_kg_m3 = (
        total_concentration_mg_l * MG_L_TO_KG_M3
    ).astype(np.float32)
    total_capacity_kg_m3 = (
        total_capacity_mg_l * MG_L_TO_KG_M3
    ).astype(np.float32)

    np.savez_compressed(
        output_dir / "hydrodynamics_truth.npz",
        time_seconds=time_seconds.astype(np.float64),
        water_surface_m=water_surface_m.astype(np.float32),
        depth_m=depth_m,
        wet_mask=wet_mask,
        velocity_reconstruction_valid=velocity_reconstruction_valid,
        velocity_x_mps=velocity_x_mps,
        velocity_y_mps=velocity_y_mps,
        face_velocity_mps=face_velocity_mps.astype(np.float32),
        face_flow_m3s=face_flow_m3s.astype(np.float32),
    )
    np.savez_compressed(
        output_dir / "sediment_truth.npz",
        time_seconds=time_seconds.astype(np.float64),
        grain_names=np.asarray(available_names, dtype="U4"),
        grain_bounds_mm=grain_bounds_mm.astype(np.float32),
        concentration_mg_l=concentration_mg_l.astype(np.float32),
        concentration_kg_m3=concentration_kg_m3,
        capacity_mg_l=capacity_mg_l.astype(np.float32),
        capacity_kg_m3=capacity_kg_m3,
        total_concentration_mg_l=total_concentration_mg_l.astype(np.float32),
        total_concentration_kg_m3=total_concentration_kg_m3,
        total_capacity_mg_l=total_capacity_mg_l.astype(np.float32),
        total_capacity_kg_m3=total_capacity_kg_m3,
        bed_shear_total_pa=bed_shear_total_pa.astype(np.float32),
        bed_shear_skin_pa=bed_shear_skin_pa.astype(np.float32),
    )
    np.savez_compressed(
        output_dir / "bed_truth.npz",
        time_seconds=time_seconds.astype(np.float64),
        initial_bed_m=initial_bed_m.astype(np.float32),
        bed_elevation_m=bed_elevation_m.astype(np.float32),
        bed_change_m=bed_change_m.astype(np.float32),
        d10_mm=d10_mm.astype(np.float32),
        d16_mm=d16_mm.astype(np.float32),
        d50_mm=d50_mm.astype(np.float32),
        d90_mm=d90_mm.astype(np.float32),
    )

    _write_cell_truth_csvs(
        output_dir=output_dir,
        time_stamps=time_stamps,
        time_seconds=time_seconds,
        coordinates_m=coordinates_m,
        water_surface_m=water_surface_m,
        depth_m=depth_m,
        wet_mask=wet_mask,
        velocity_reconstruction_valid=velocity_reconstruction_valid,
        velocity_x_mps=velocity_x_mps,
        velocity_y_mps=velocity_y_mps,
        available_names=available_names,
        concentration_kg_m3=concentration_kg_m3,
        capacity_kg_m3=capacity_kg_m3,
        total_concentration_kg_m3=total_concentration_kg_m3,
        total_capacity_kg_m3=total_capacity_kg_m3,
        bed_shear_total_pa=bed_shear_total_pa,
        bed_shear_skin_pa=bed_shear_skin_pa,
        bed_elevation_m=bed_elevation_m,
        bed_change_m=bed_change_m,
        d10_mm=d10_mm,
        d16_mm=d16_mm,
        d50_mm=d50_mm,
        d90_mm=d90_mm,
    )

    metadata = {
        "source": str(source),
        "area": AREA,
        "time_count": int(time_days.size),
        "cell_count": int(coordinates_m.shape[0]),
        "face_count": int(face_velocity_mps.shape[1]),
        "start_time": str(time_stamps[0]),
        "end_time": str(time_stamps[-1]),
        "time_interval_seconds": float(np.median(np.diff(time_seconds))),
        "grain_names": available_names,
        "grain_bounds_columns": ["lower_mm", "geometric_mean_mm", "upper_mm"],
        "array_layout": {
            "cell_fields": "[time, cell]",
            "grain_fields": "[time, cell, grain]",
            "face_fields": "[time, face]",
        },
        "csv_layout": {
            "hydrodynamics_truth.csv": "one row per time and cell",
            "sediment_truth.csv": "one row per time and cell; grain classes are columns",
            "bed_truth.csv": "one row per time and cell",
        },
        "derived_fields": {
            "depth_m": "max(water_surface_m - bed_elevation_m, 0)",
            "velocity_x_mps": (
                "Least-squares reconstruction from signed face-normal velocity"
            ),
            "velocity_y_mps": (
                "Least-squares reconstruction from signed face-normal velocity"
            ),
            "velocity_reconstruction_valid": (
                "False where adjacent face normals cannot uniquely reconstruct 2D velocity"
            ),
        },
        "unit_conversions": {
            "ft_to_m": FT_TO_M,
            "cfs_to_m3s": CFS_TO_M3S,
            "mg_l_to_kg_m3": MG_L_TO_KG_M3,
            "psf_to_pa": PSF_TO_PA,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Extracted HEC-RAS truth to: {output_dir}")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}: {path.stat().st_size / 1024 / 1024:.2f} MiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/Chippewa_2D/Chippewa_2D.p02.hdf"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/Chippewa_2D/validation_truth"),
    )
    args = parser.parse_args()
    extract(args.source, args.output_dir)


if __name__ == "__main__":
    main()
