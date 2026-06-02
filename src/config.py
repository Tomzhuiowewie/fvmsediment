# config.py - YAML 配置读取

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# epsilon 常量
EPS_SAFE = 1e-6
EPS_DIVISION = 1e-8
EPS_VELOCITY_CLAMP = 1e-3


@dataclass
class SimulationConfig:
    bounds: Dict[str, float]
    bbox: Dict[str, float]
    resolution: float
    n_gauss_points: int
    include_time_terms: bool
    simulation_time: float
    n_windows: int
    sample_dt: float
    window_dt: float
    output_dt: float
    typical_depth: float
    typical_velocity: float
    g: float
    n_manning: float
    grain_diameters: List[float]
    beta_default: float
    epsilon_default: float
    sediment_residual_scale: float
    adaptation_length: float
    rho_s: float
    rho_w: float
    kinematic_viscosity: float
    wu_theta_cr: float
    skin_shear_factor: float
    alpha_active_layer: float
    w_capacity: float
    w_initial_sediment: float
    initial_sediment_concentration: List[float]
    w_inlet_sediment: float
    inlet_sediment_concentration: List[float]
    source_sharpness: float
    porosity: float
    bed_slope_coefficient: float
    min_bed_elevation: Optional[float]
    bc_default: Dict[str, Any]
    training: Dict[str, Any]

    @property
    def num_grain_classes(self) -> int:
        return len(self.grain_diameters)


def load_config(path) -> SimulationConfig:
    yaml_path = Path(path)
    with yaml_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    domain = data['domain']
    fvm = data['fvm']
    physics = data['physics']
    boundary = data['boundary']
    training = data['training']
    flow = physics['flow']
    sediment = physics['sediment']
    morphodynamics = physics.get('morphodynamics', {})
    typical_velocity = _resolve_typical_velocity(flow.get('typical_velocity', 'auto'), boundary)

    return SimulationConfig(
        bounds=domain['bounds'],
        bbox=domain['bbox'],
        resolution=float(domain['resolution']),
        n_gauss_points=int(fvm['n_gauss_points']),
        include_time_terms=bool(physics['include_time_terms']),
        simulation_time=float(physics['simulation_time']),
        n_windows=int(physics['n_windows']),
        sample_dt=float(physics['sample_dt']),
        window_dt=float(physics['window_dt']),
        output_dt=float(physics['output_dt']),
        typical_depth=float(flow['typical_depth']),
        typical_velocity=typical_velocity,
        g=float(flow.get('g', 9.81)),
        n_manning=float(flow.get('n_manning', 0.01)),
        grain_diameters=list(sediment['grain_diameters']),
        beta_default=float(sediment.get('beta_default', 1.0)),
        epsilon_default=float(sediment.get('epsilon_default', 0.1)),
        sediment_residual_scale=float(sediment.get('residual_scale', 1.0)),
        adaptation_length=float(sediment.get('adaptation_length', 50.0)),
        rho_s=float(sediment.get('rho_s', 2650.0)),
        rho_w=float(sediment.get('rho_w', 1000.0)),
        kinematic_viscosity=float(sediment.get('kinematic_viscosity', 1.0e-6)),
        wu_theta_cr=float(sediment.get('wu_theta_cr', 0.03)),
        skin_shear_factor=float(sediment.get('skin_shear_factor', 1.0)),
        alpha_active_layer=float(sediment.get('alpha_active_layer', 10.0)),
        w_capacity=float(sediment.get('w_capacity', 0.05)),
        w_initial_sediment=float(sediment.get('w_initial_sediment', 1.0)),
        initial_sediment_concentration=_resolve_initial_sediment_concentration(
            sediment.get('initial_concentration', 0.0),
            len(sediment['grain_diameters']),
        ),
        w_inlet_sediment=float(sediment.get('w_inlet_sediment', 1.0)),
        inlet_sediment_concentration=_resolve_initial_sediment_concentration(
            sediment.get('inlet_concentration', sediment.get('initial_concentration', 0.0)),
            len(sediment['grain_diameters']),
        ),
        source_sharpness=float(sediment.get('source_sharpness', EPS_VELOCITY_CLAMP)),
        porosity=float(morphodynamics.get('porosity', 0.4)),
        bed_slope_coefficient=float(morphodynamics.get('bed_slope_coefficient', 0.2)),
        min_bed_elevation=(
            None
            if morphodynamics.get('min_bed_elevation') is None
            else float(morphodynamics['min_bed_elevation'])
        ),
        bc_default=dict(boundary),
        training=dict(training),
    )


def _resolve_typical_velocity(value, boundary: Dict[str, Any]) -> float:
    """Resolve velocity scale, optionally tying it to the initial/boundary velocity."""
    if isinstance(value, str) and value.strip().lower() == 'auto':
        u0 = float(boundary.get('u', 0.0))
        v0 = float(boundary.get('v', 0.0))
        initial_speed = (u0 ** 2 + v0 ** 2) ** 0.5
        return max(1.2 * initial_speed, EPS_VELOCITY_CLAMP)
    return max(float(value), EPS_VELOCITY_CLAMP)


def _resolve_initial_sediment_concentration(value, n_grains: int) -> List[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * n_grains
    values = [float(v) for v in value]
    if len(values) != n_grains:
        raise ValueError("initial_concentration 的长度必须与 grain_diameters 一致。")
    return values
