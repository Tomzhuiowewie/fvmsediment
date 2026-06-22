#!/usr/bin/env python3
"""Export a trained PINN on the HEC-RAS validation cells and time axis."""

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import map_coordinates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import FT_TO_M, load_real_case_data
from src.model import FlowPINN, SedimentPINN
from src.physics import ClosureFormulation
from src.utils import build_xyt


GRAIN_NAMES = ["FS", "MS", "CS", "VCS", "VFG", "FG", "MG", "CG", "VCG", "SC"]


def _latest_file(directory, pattern):
    files = sorted(Path(directory).glob(pattern), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No file matching {pattern!r} in {directory}")
    return files[-1]


def _resolve_paths(args):
    checkpoint = args.checkpoint or _latest_file(args.output_dir, "final_checkpoint_*.pt")
    timestamp = checkpoint.stem.removeprefix("final_checkpoint_")
    config = args.config or args.output_dir / f"config_used_{timestamp}.yaml"
    if not config.exists():
        raise FileNotFoundError(config)
    return checkpoint, config, timestamp


def _load_truth_axes(truth_dir):
    cells = np.genfromtxt(
        truth_dir / "cell_geometry.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    times = np.genfromtxt(
        truth_dir / "time.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    return cells, times


def _hecras_to_local_xy(cells, dem_path):
    with Image.open(dem_path) as image:
        width, height = image.size
        tiepoint = image.tag_v2.get(33922)
        pixel_scale = image.tag_v2.get(33550)
    if tiepoint is None or pixel_scale is None:
        raise ValueError("DEM GeoTIFF is missing tiepoint or pixel scale metadata.")
    origin_x_ft = float(tiepoint[3])
    origin_y_top_ft = float(tiepoint[4])
    dx_ft = float(pixel_scale[0])
    dy_ft = float(pixel_scale[1])
    origin_y_bottom_ft = origin_y_top_ft - height * dy_ft
    local_x = (np.asarray(cells["x_ft"], dtype=np.float64) - origin_x_ft) * FT_TO_M
    local_y = (
        np.asarray(cells["y_ft"], dtype=np.float64) - origin_y_bottom_ft
    ) * FT_TO_M
    return np.column_stack([local_x, local_y]), {
        "origin_x_ft": origin_x_ft,
        "origin_y_top_ft": origin_y_top_ft,
        "origin_y_bottom_ft": origin_y_bottom_ft,
        "pixel_dx_ft": dx_ft,
        "pixel_dy_ft": dy_ft,
        "raster_width": width,
        "raster_height": height,
    }


def _infer_architecture(state_dict):
    hidden_dim, input_dim = state_dict["input_layer.0.weight"].shape
    output_dim = state_dict["output_layer.2.weight"].shape[0]
    block_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("res_blocks.") and key.endswith(".fc1.weight")
    }
    return input_dim, hidden_dim, len(block_ids), output_dim


def _create_models(checkpoint, cfg, device):
    flow_state = checkpoint["flow_model"]
    sediment_state = checkpoint["sediment_model"]
    flow_dims = _infer_architecture(flow_state)
    sediment_dims = _infer_architecture(sediment_state)
    configured_grains = len(getattr(cfg, "grain_diameters", []) or [])
    if configured_grains and sediment_dims[3] in (configured_grains + 1, 2 * configured_grains):
        n_grains = configured_grains
    elif sediment_dims[3] % 2 == 0:
        n_grains = sediment_dims[3] // 2
    else:
        n_grains = sediment_dims[3] - 1

    flow_model = FlowPINN(*flow_dims).to(device)
    sediment_model = SedimentPINN(
        input_dim=sediment_dims[0],
        hidden_dim=sediment_dims[1],
        num_block=sediment_dims[2],
        output_dim=sediment_dims[3],
        n_concentration_outputs=n_grains,
        initial_concentration=cfg.initial_sediment_concentration,
        bed_change_scale=cfg.bed_change_scale,
    ).to(device)
    flow_model.load_state_dict(flow_state)
    sediment_model.load_state_dict(sediment_state)
    flow_model.eval()
    sediment_model.eval()
    return flow_model, sediment_model, n_grains


def _sample_grid_fields(fields, local_xy, resolution):
    """Bilinearly sample [field, y, x] arrays at target local coordinates."""
    row = local_xy[:, 1] / resolution - 0.5
    col = local_xy[:, 0] / resolution - 0.5
    coords = np.vstack([row, col])
    return np.stack([
        map_coordinates(field, coords, order=1, mode="nearest")
        for field in fields
    ])


def _interpolate_final_gradation(checkpoint, local_xy, real_case, n_grains):
    fractions = np.asarray(checkpoint["active_layer_frac"], dtype=np.float32)
    fraction_grid = fractions.T.reshape(
        n_grains,
        real_case.bed_grid.shape[0],
        real_case.bed_grid.shape[1],
    )
    sampled = _sample_grid_fields(fraction_grid, local_xy, real_case.resolution).T
    sampled = np.clip(sampled, 1.0e-8, None)
    return sampled / sampled.sum(axis=1, keepdims=True)


def _percentile_diameters(fractions, upper_bounds_mm, percentiles):
    cumulative = np.cumsum(fractions, axis=-1)
    lower_bounds = np.concatenate([[upper_bounds_mm[0] / 2.0], upper_bounds_mm[:-1]])
    result = np.empty(fractions.shape[:-1] + (len(percentiles),), dtype=np.float32)
    flat_fractions = fractions.reshape(-1, fractions.shape[-1])
    flat_cumulative = cumulative.reshape(-1, cumulative.shape[-1])
    flat_result = result.reshape(-1, len(percentiles))
    for row_id, (row_fraction, row_cumulative) in enumerate(
        zip(flat_fractions, flat_cumulative)
    ):
        previous_cumulative = 0.0
        for p_id, percentile in enumerate(percentiles):
            grain_id = int(np.searchsorted(row_cumulative, percentile, side="left"))
            grain_id = min(grain_id, row_fraction.size - 1)
            previous_cumulative = (
                row_cumulative[grain_id - 1] if grain_id > 0 else 0.0
            )
            denominator = max(
                row_cumulative[grain_id] - previous_cumulative,
                1.0e-12,
            )
            fraction = np.clip(
                (percentile - previous_cumulative) / denominator,
                0.0,
                1.0,
            )
            low = max(lower_bounds[grain_id], 1.0e-12)
            high = max(upper_bounds_mm[grain_id], low)
            flat_result[row_id, p_id] = np.exp(
                np.log(low) + fraction * (np.log(high) - np.log(low))
            )
    return result


def _write_csvs(output_dir, times, cells, hydro, sediment, bed, grain_names):
    output_dir.mkdir(parents=True, exist_ok=True)
    common = ["time_id", "timestamp", "time_seconds", "cell_id", "x_m", "y_m"]
    n_time, n_cells = hydro["depth_m"].shape

    with (output_dir / "pinn_hydrodynamics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(common + [
            "water_surface_m", "depth_m", "wet",
            "velocity_x_mps", "velocity_y_mps",
        ])
        for time_id in range(n_time):
            for cell_id in range(n_cells):
                writer.writerow([
                    time_id, times["timestamp"][time_id], times["time_seconds"][time_id],
                    cell_id, cells["x_m"][cell_id], cells["y_m"][cell_id],
                    hydro["water_surface_m"][time_id, cell_id],
                    hydro["depth_m"][time_id, cell_id],
                    int(hydro["wet"][time_id, cell_id]),
                    hydro["velocity_x_mps"][time_id, cell_id],
                    hydro["velocity_y_mps"][time_id, cell_id],
                ])

    with (output_dir / "pinn_sediment.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(common + [
            *[f"concentration_{name}_kg_m3" for name in grain_names],
            "total_concentration_kg_m3",
            *[f"capacity_{name}_kg_m3" for name in grain_names],
            "total_capacity_kg_m3", "bed_shear_total_pa", "bed_shear_skin_pa",
        ])
        for time_id in range(n_time):
            for cell_id in range(n_cells):
                writer.writerow([
                    time_id, times["timestamp"][time_id], times["time_seconds"][time_id],
                    cell_id, cells["x_m"][cell_id], cells["y_m"][cell_id],
                    *sediment["concentration_kg_m3"][time_id, cell_id],
                    sediment["total_concentration_kg_m3"][time_id, cell_id],
                    *sediment["capacity_kg_m3"][time_id, cell_id],
                    sediment["total_capacity_kg_m3"][time_id, cell_id],
                    sediment["bed_shear_total_pa"][time_id, cell_id],
                    sediment["bed_shear_skin_pa"][time_id, cell_id],
                ])

    with (output_dir / "pinn_bed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(common + [
            "bed_elevation_m", "bed_change_m",
            "d10_mm", "d16_mm", "d50_mm", "d90_mm",
        ])
        for time_id in range(n_time):
            for cell_id in range(n_cells):
                writer.writerow([
                    time_id, times["timestamp"][time_id], times["time_seconds"][time_id],
                    cell_id, cells["x_m"][cell_id], cells["y_m"][cell_id],
                    bed["bed_elevation_m"][time_id, cell_id],
                    bed["bed_change_m"][time_id, cell_id],
                    *bed["percentile_diameters_mm"][time_id, cell_id],
                ])


def export(args):
    checkpoint_path, config_path, timestamp = _resolve_paths(args)
    cfg = load_config(config_path)
    real_case = load_real_case_data(cfg.data)
    cells, times = _load_truth_axes(args.truth_dir)
    local_xy, transform = _hecras_to_local_xy(cells, cfg.data["dem_path"])
    target_times = np.asarray(times["time_seconds"], dtype=np.float64)
    device = torch.device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    flow_model, sediment_model, n_grains = _create_models(checkpoint, cfg, device)
    if n_grains != len(cfg.grain_diameters):
        raise ValueError("Checkpoint sediment outputs do not match config grain count.")
    grain_names = GRAIN_NAMES[:n_grains]

    initial_bed = _sample_grid_fields(
        real_case.bed_grid[None, :, :],
        local_xy,
        real_case.resolution,
    )[0].astype(np.float32)
    final_gradation = _interpolate_final_gradation(
        checkpoint,
        local_xy,
        real_case,
        n_grains,
    )
    initial_gradation = np.tile(
        np.asarray(real_case.grain_fractions, dtype=np.float32),
        (local_xy.shape[0], 1),
    )
    if initial_gradation.shape[1] != n_grains:
        raise ValueError("Initial gradation does not match checkpoint grain count.")

    n_time = target_times.size
    n_cells = local_xy.shape[0]
    h = np.empty((n_time, n_cells), dtype=np.float32)
    u = np.empty_like(h)
    v = np.empty_like(h)
    concentration = np.empty((n_time, n_cells, n_grains), dtype=np.float32)
    capacity = np.empty_like(concentration)
    tau_total = np.empty_like(h)
    tau_skin = np.empty_like(h)
    bed_change = np.empty_like(h)
    percentiles = np.empty((n_time, n_cells, 4), dtype=np.float32)

    coords = torch.as_tensor(local_xy, dtype=torch.float32, device=device)
    closure = ClosureFormulation(SimpleNamespace(
        rho_w=cfg.rho_w,
        rho_s=cfg.rho_s,
        g=cfg.g,
        n_manning=cfg.n_manning,
        kinematic_viscosity=cfg.kinematic_viscosity,
        wu_theta_cr=cfg.wu_theta_cr,
        skin_shear_factor=cfg.skin_shear_factor,
    ))
    grain_diameters = torch.as_tensor(
        cfg.grain_diameters,
        dtype=torch.float32,
        device=device,
    )
    # config stores the HEC-RAS geometric-mean representative diameter.
    # The FS-SC class bounds progress by a factor of two, so d_upper=d_rep*sqrt(2).
    upper_bounds_mm = (
        np.asarray(cfg.grain_diameters, dtype=np.float64)
        * np.sqrt(2.0)
        * 1000.0
    )

    for time_id, time_seconds in enumerate(target_times):
        t_norm = float(time_seconds / cfg.simulation_time)
        xyt = build_xyt(coords, t_norm, real_case.bounds, device, requires_grad=False)
        with torch.no_grad():
            h_t, u_t, v_t = FlowPINN.decode_output(
                flow_model(xyt),
                cfg.typical_depth,
                cfg.typical_velocity,
            )
            sediment_raw = sediment_model(xyt)
            c_t = sediment_raw[:, :n_grains]
            dzb_raw = sediment_raw[:, n_grains:]
            if dzb_raw.shape[1] >= n_grains:
                dzb_t = torch.sum(dzb_raw[:, :n_grains], dim=1)
            elif dzb_raw.shape[1] == 1:
                dzb_t = dzb_raw[:, 0]
            else:
                dzb_t = torch.zeros_like(c_t[:, 0])
            p_t_np = (
                initial_gradation
                + t_norm * (final_gradation - initial_gradation)
            )
            p_t_np = np.clip(p_t_np, 1.0e-8, None)
            p_t_np /= p_t_np.sum(axis=1, keepdims=True)
            p_t = torch.as_tensor(p_t_np, dtype=torch.float32, device=device)
            _, capacity_t, diagnostics = closure.TransportPotential_Wu(
                h_t,
                u_t,
                v_t,
                grain_diameters,
                p_t,
            )
        h[time_id] = h_t.squeeze(1).cpu().numpy()
        u[time_id] = u_t.squeeze(1).cpu().numpy()
        v[time_id] = v_t.squeeze(1).cpu().numpy()
        concentration[time_id] = c_t.cpu().numpy() * cfg.rho_s
        capacity[time_id] = capacity_t.cpu().numpy() * cfg.rho_s
        tau_total[time_id] = diagnostics["tau_b"].squeeze(1).cpu().numpy()
        tau_skin[time_id] = diagnostics["tau_skin"].squeeze(1).cpu().numpy()
        bed_change[time_id] = dzb_t.cpu().numpy()
        percentiles[time_id] = _percentile_diameters(
            p_t_np,
            upper_bounds_mm,
            [0.10, 0.16, 0.50, 0.90],
        )
        if time_id % 100 == 0 or time_id == n_time - 1:
            print(f"Inference {time_id + 1}/{n_time}")

    bed_elevation = initial_bed[None, :] + bed_change
    water_surface = h + bed_elevation
    hydro = {
        "water_surface_m": water_surface,
        "depth_m": h,
        "wet": h > 1.0e-5,
        "velocity_x_mps": u,
        "velocity_y_mps": v,
    }
    sediment = {
        "concentration_kg_m3": concentration,
        "total_concentration_kg_m3": concentration.sum(axis=2),
        "capacity_kg_m3": capacity,
        "total_capacity_kg_m3": capacity.sum(axis=2),
        "bed_shear_total_pa": tau_total,
        "bed_shear_skin_pa": tau_skin,
    }
    bed = {
        "bed_elevation_m": bed_elevation,
        "bed_change_m": bed_change,
        "percentile_diameters_mm": percentiles,
    }

    output_dir = args.export_dir or args.output_dir / f"pinn_inference_{timestamp}"
    _write_csvs(output_dir, times, cells, hydro, sediment, bed, grain_names)
    np.savez_compressed(output_dir / "pinn_hydrodynamics.npz", **hydro)
    np.savez_compressed(output_dir / "pinn_sediment.npz", **sediment)
    np.savez_compressed(output_dir / "pinn_bed.npz", **bed)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "truth_grid": str(args.truth_dir),
        "timestamp": timestamp,
        "device": str(device),
        "time_count": n_time,
        "cell_count": n_cells,
        "grain_names": grain_names,
        "coordinate_transform": transform,
        "field_methods": {
            "h_u_v_concentration": "PINN direct evaluation at HEC-RAS cell centers",
            "bed_elevation": (
                "bilinear interpolation of the initial DEM to HEC-RAS cell centers "
                "plus the sum of SedimentPINN per-grain cumulative bed-change outputs"
            ),
            "bed_change": "sum of SedimentPINN per-grain bed-change outputs evaluated directly at each x/y/t",
            "capacity_and_shear": "Wu closure evaluated from PINN h/u/v and interpolated gradation",
            "sediment_units": (
                "PINN volumetric concentration and capacity multiplied by sediment "
                "density rho_s to obtain kg/m3 for HEC-RAS comparison"
            ),
            "gradation": (
                "linear transition from initial uniform spatial gradation to the "
                "checkpoint final active-layer gradation because intermediate "
                "gradation history was not saved"
            ),
            "grain_diameters": (
                "config values are HEC-RAS geometric-mean representative diameters; "
                "class upper bounds used for D-percentiles are representative "
                "diameter times sqrt(2)"
            ),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"PINN inference exported to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs-autodl"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=Path("data/Chippewa_2D/validation_truth"),
    )
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    export(parser.parse_args())


if __name__ == "__main__":
    main()
