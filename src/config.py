# config.py - YAML 配置读取
# YAML 文件保持按 data/fvm/physics/training 分组；Python 侧也使用分组 dataclass。
# 为兼容现有代码，SimulationConfig 继续提供旧的扁平属性访问。

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


# epsilon 常量
EPS_SAFE = 1e-6
EPS_DIVISION = 1e-8
EPS_VELOCITY_CLAMP = 1e-3


@dataclass
class FlowConfig:
    typical_depth: float
    typical_velocity: float
    g: float
    n_manning: float
    adaptive_loss_weighting: bool
    adaptive_weight_ema_decay: float
    adaptive_weight_min: float
    adaptive_weight_max: float


@dataclass
class SedimentConfig:
    grain_diameters: List[float]
    beta_default: float
    epsilon_default: float
    residual_scale: float
    adaptation_length: float
    rho_s: float
    rho_w: float
    kinematic_viscosity: float
    wu_theta_cr: float
    skin_shear_factor: float
    alpha_active_layer: float
    w_capacity: float
    w_initial_sediment: float
    initial_concentration: List[float]
    w_inlet_sediment: float
    source_sharpness: float


@dataclass
class MorphodynamicsConfig:
    porosity: float
    active_layer_thickness: float
    bed_slope_coefficient: float
    bed_change_scale: float
    exchange_weight: float
    bed_slope_diffusion_weight: float


@dataclass
class TimeConfig:
    include_time_terms: bool
    simulation_time: float
    sample_dt: float
    output_dt: float


@dataclass
class TrainingConfig:
    values: Dict[str, Any]

    def get(self, key, default=None):
        return self.values.get(key, default)

    def __getitem__(self, key):
        return self.values[key]

    def __contains__(self, key):
        return key in self.values

    def items(self):
        return self.values.items()

    def keys(self):
        return self.values.keys()

    @property
    def flow_epochs(self) -> int:
        return int(self.values['flow_epochs'])

    @property
    def sediment_epochs(self) -> int:
        return int(self.values['sediment_epochs'])

    @property
    def joint_epochs(self) -> int:
        return int(self.values['joint_epochs'])

    @property
    def output_dir(self) -> str:
        return self.values.get('output_dir', 'outputs')

    @property
    def checkpoint_dir(self) -> str:
        return self.values.get('checkpoint_dir', f"{self.output_dir}/checkpoints")


@dataclass
class SimulationConfig:
    data: Dict[str, Any]
    n_gauss_points: int
    time: TimeConfig
    flow: FlowConfig
    sediment: SedimentConfig
    morphodynamics: MorphodynamicsConfig
    training: TrainingConfig

    @property
    def num_grain_classes(self) -> int:
        return len(self.sediment.grain_diameters)

    # ---- Backward-compatible flat accessors ----
    @property
    def include_time_terms(self): return self.time.include_time_terms
    @property
    def simulation_time(self): return self.time.simulation_time
    @property
    def sample_dt(self): return self.time.sample_dt
    @property
    def output_dt(self): return self.time.output_dt

    @property
    def typical_depth(self): return self.flow.typical_depth
    @property
    def typical_velocity(self): return self.flow.typical_velocity
    @property
    def g(self): return self.flow.g
    @property
    def n_manning(self): return self.flow.n_manning
    @property
    def adaptive_flow_weighting(self): return self.flow.adaptive_loss_weighting
    @property
    def flow_weight_ema_decay(self): return self.flow.adaptive_weight_ema_decay
    @property
    def flow_weight_min(self): return self.flow.adaptive_weight_min
    @property
    def flow_weight_max(self): return self.flow.adaptive_weight_max

    @property
    def grain_diameters(self): return self.sediment.grain_diameters
    @property
    def beta_default(self): return self.sediment.beta_default
    @property
    def epsilon_default(self): return self.sediment.epsilon_default
    @property
    def sediment_residual_scale(self): return self.sediment.residual_scale
    @property
    def adaptation_length(self): return self.sediment.adaptation_length
    @property
    def rho_s(self): return self.sediment.rho_s
    @property
    def rho_w(self): return self.sediment.rho_w
    @property
    def kinematic_viscosity(self): return self.sediment.kinematic_viscosity
    @property
    def wu_theta_cr(self): return self.sediment.wu_theta_cr
    @property
    def skin_shear_factor(self): return self.sediment.skin_shear_factor
    @property
    def alpha_active_layer(self): return self.sediment.alpha_active_layer
    @property
    def w_capacity(self): return self.sediment.w_capacity
    @property
    def w_initial_sediment(self): return self.sediment.w_initial_sediment
    @property
    def initial_sediment_concentration(self): return self.sediment.initial_concentration
    @property
    def w_inlet_sediment(self): return self.sediment.w_inlet_sediment
    @property
    def source_sharpness(self): return self.sediment.source_sharpness

    @property
    def porosity(self): return self.morphodynamics.porosity
    @property
    def active_layer_thickness(self): return self.morphodynamics.active_layer_thickness
    @property
    def bed_slope_coefficient(self): return self.morphodynamics.bed_slope_coefficient
    @property
    def bed_change_scale(self): return self.morphodynamics.bed_change_scale
    @property
    def exchange_weight(self): return self.morphodynamics.exchange_weight
    @property
    def bed_slope_diffusion_weight(self): return self.morphodynamics.bed_slope_diffusion_weight


def load_config(path) -> SimulationConfig:
    yaml_path = Path(path)
    with yaml_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    fvm = data['fvm']
    physics = data['physics']
    data_settings = data['data']
    flow_raw = physics['flow']
    sediment_raw = physics['sediment']
    morph_raw = physics.get('morphodynamics', {})
    training_raw = data['training']

    typical_velocity = max(float(flow_raw['typical_velocity']), EPS_VELOCITY_CLAMP)
    grain_diameters = list(sediment_raw.get('grain_diameters') or [])
    initial_concentration = _resolve_initial_sediment_concentration(
        sediment_raw.get('initial_concentration', 0.0),
        len(grain_diameters),
    )

    return SimulationConfig(
        data=dict(data_settings),
        n_gauss_points=int(fvm['n_gauss_points']),
        time=TimeConfig(
            include_time_terms=bool(physics['include_time_terms']),
            simulation_time=float(physics['simulation_time']),
            sample_dt=float(physics['sample_dt']),
            output_dt=float(physics['output_dt']),
        ),
        flow=FlowConfig(
            typical_depth=float(flow_raw['typical_depth']),
            typical_velocity=typical_velocity,
            g=float(flow_raw.get('g', 9.81)),
            n_manning=float(flow_raw.get('n_manning', 0.01)),
            adaptive_loss_weighting=bool(flow_raw.get('adaptive_loss_weighting', True)),
            adaptive_weight_ema_decay=float(flow_raw.get('adaptive_weight_ema_decay', 0.95)),
            adaptive_weight_min=float(flow_raw.get('adaptive_weight_min', 0.05)),
            adaptive_weight_max=float(flow_raw.get('adaptive_weight_max', 20.0)),
        ),
        sediment=SedimentConfig(
            grain_diameters=grain_diameters,
            beta_default=float(sediment_raw.get('beta_default', 1.0)),
            epsilon_default=float(sediment_raw.get('epsilon_default', 0.1)),
            residual_scale=float(sediment_raw.get('residual_scale', 1.0)),
            adaptation_length=float(sediment_raw.get('adaptation_length', 50.0)),
            rho_s=float(sediment_raw.get('rho_s', 2650.0)),
            rho_w=float(sediment_raw.get('rho_w', 1000.0)),
            kinematic_viscosity=float(sediment_raw.get('kinematic_viscosity', 1.0e-6)),
            wu_theta_cr=float(sediment_raw.get('wu_theta_cr', 0.03)),
            skin_shear_factor=float(sediment_raw.get('skin_shear_factor', 1.0)),
            alpha_active_layer=float(sediment_raw.get('alpha_active_layer', 10.0)),
            w_capacity=float(sediment_raw.get('w_capacity', 0.05)),
            w_initial_sediment=float(sediment_raw.get('w_initial_sediment', 1.0)),
            initial_concentration=initial_concentration,
            w_inlet_sediment=float(sediment_raw.get('w_inlet_sediment', 1.0)),
            source_sharpness=float(sediment_raw.get('source_sharpness', EPS_VELOCITY_CLAMP)),
        ),
        morphodynamics=MorphodynamicsConfig(
            porosity=float(morph_raw.get('porosity', 0.4)),
            active_layer_thickness=float(morph_raw.get('active_layer_thickness', 0.5)),
            bed_slope_coefficient=float(morph_raw.get('bed_slope_coefficient', 0.2)),
            bed_change_scale=float(morph_raw.get('bed_change_scale', 0.1)),
            exchange_weight=float(morph_raw.get('exchange_weight', 1.0)),
            bed_slope_diffusion_weight=float(morph_raw.get('bed_slope_diffusion_weight', 1.0)),
        ),
        training=TrainingConfig(dict(training_raw)),
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
