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
    data: Dict[str, Any]
    n_gauss_points: int
    include_time_terms: bool
    simulation_time: float
    sample_dt: float
    morph_dt: float
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
    source_sharpness: float
    porosity: float
    active_layer_thickness: float
    bed_slope_coefficient: float
    bed_change_scale: float
    exchange_weight: float
    bed_slope_diffusion_weight: float
    training: Dict[str, Any]

    @property
    def num_grain_classes(self) -> int:
        return len(self.grain_diameters)


def load_config(path) -> SimulationConfig:
    yaml_path = Path(path)
    with yaml_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    fvm = data['fvm']
    physics = data['physics']
    training = data['training']
    data_settings = data['data']
    flow = physics['flow']
    sediment = physics['sediment']
    morphodynamics = physics.get('morphodynamics', {})
    typical_velocity = max(float(flow['typical_velocity']), EPS_VELOCITY_CLAMP)
    grain_diameters = list(sediment.get('grain_diameters') or [])

    return SimulationConfig(
        data=dict(data_settings),
        n_gauss_points=int(fvm['n_gauss_points']),
        include_time_terms=bool(physics['include_time_terms']),
        simulation_time=float(physics['simulation_time']),
        sample_dt=float(physics['sample_dt']),
        morph_dt=float(physics.get('morph_dt', physics.get('output_dt', physics['sample_dt']))),
        output_dt=float(physics['output_dt']),
        typical_depth=float(flow['typical_depth']),
        typical_velocity=typical_velocity,
        g=float(flow.get('g', 9.81)),
        n_manning=float(flow.get('n_manning', 0.01)),
        grain_diameters=grain_diameters,
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
            len(grain_diameters),
        ),
        w_inlet_sediment=float(sediment.get('w_inlet_sediment', 1.0)),
        source_sharpness=float(sediment.get('source_sharpness', EPS_VELOCITY_CLAMP)),
        porosity=float(morphodynamics.get('porosity', 0.4)),
        active_layer_thickness=float(morphodynamics.get('active_layer_thickness', 0.5)),
        bed_slope_coefficient=float(morphodynamics.get('bed_slope_coefficient', 0.2)),
        bed_change_scale=float(morphodynamics.get('bed_change_scale', 0.1)),
        exchange_weight=float(morphodynamics.get('exchange_weight', 1.0)),
        bed_slope_diffusion_weight=float(morphodynamics.get('bed_slope_diffusion_weight', 1.0)),
        training=dict(training),
    )
def _resolve_initial_sediment_concentration(value, n_grains: int) -> List[float]:
    if n_grains == 0:
        return []
    if isinstance(value, (int, float)):
        return [float(value)] * n_grains
    values = [float(v) for v in value]
    if len(values) != n_grains:
        raise ValueError("initial_concentration 的长度必须与 grain_diameters 一致。")
    return values
