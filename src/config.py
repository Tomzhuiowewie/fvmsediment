# config.py - YAML 配置读取

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
    source_sharpness: float
    porosity: float
    bed_slope_coefficient: float
    bc_default: Dict[str, Any]
    training: Dict[str, Any]

    @property
    def num_grain_classes(self) -> int:
        return len(self.grain_diameters)


def load_config(path) -> SimulationConfig:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
    with yaml_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件必须是 YAML 字典结构。")

    domain = data['domain']
    fvm = data['fvm']
    physics = data['physics']
    boundary = data['boundary']
    training = data['training']
    flow = physics['flow']
    sediment = physics['sediment']
    morphodynamics = physics.get('morphodynamics', {})

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
        typical_velocity=float(flow['typical_velocity']),
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
        source_sharpness=float(sediment.get('source_sharpness', EPS_VELOCITY_CLAMP)),
        porosity=float(morphodynamics.get('porosity', 0.4)),
        bed_slope_coefficient=float(morphodynamics.get('bed_slope_coefficient', 0.2)),
        bc_default=dict(boundary),
        training=dict(training),
    )
