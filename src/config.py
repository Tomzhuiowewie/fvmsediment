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
    ag: float
    grain_diameters: List[float]
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
        typical_depth=float(physics['flow']['typical_depth']),
        typical_velocity=float(physics['flow']['typical_velocity']),
        ag=float(physics['sediment']['ag']),
        grain_diameters=list(physics['sediment']['grain_diameters']),
        bc_default=dict(boundary),
        training=dict(training),
    )