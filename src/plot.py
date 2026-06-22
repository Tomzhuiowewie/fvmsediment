# plot.py - 训练结果可视化
# 输出床面等值图、中心剖面、训练损失历史和自适应权重曲线。

import os
from io import BytesIO

import numpy as np

from .config import EPS_SAFE

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False


def visualize_results(
    mesh,
    bed_history,
    bbox,
    resolution,
    history,
    simulation_time,
    case_name='default',
    output_dir=None,
    run_timestamp=None,
):
    if not HAS_MATPLOTLIB:
        print("未安装 matplotlib，跳过绘图。")
        return

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    plot_context = _prepare_plot_context(mesh, bed_history, bbox, resolution, history, simulation_time)
    save_path = _make_save_path(case_name, output_dir, run_timestamp)

    plot_bed_evolution(bed_history, plot_context, save_path('bed'))
    plot_bed_change_evolution(bed_history, plot_context, save_path('bed_change'))
    plot_bed_profiles(bed_history, plot_context, save_path('profiles'))
    plot_training_history(history, plot_context, simulation_time, save_path('losses'))
    plot_stage_loss_breakdown(history, save_path)
    plot_flow_adaptive_weights(history, save_path('flow_weights'))

    active_mask = plot_context.get('active_mask')
    z0 = _mask_bed(bed_history[0].reshape(plot_context['ny'], plot_context['nx']), active_mask)
    z1 = _mask_bed(bed_history[-1].reshape(plot_context['ny'], plot_context['nx']), active_mask)
    initial_peak = float(np.nanmax(z0))
    final_peak = float(np.nanmax(z1))
    dz = final_peak - initial_peak
    print(f'\n  Initial river peak: {initial_peak:.4f}m  Final river peak: {final_peak:.4f}m')
    print(f'  Peak change: {dz:+.4f}m ({dz / max(initial_peak, EPS_SAFE) * 100:.1f}%)')


def _make_save_path(case_name, output_dir, run_timestamp=None):
    def _save_path(name):
        suffix = f'_{run_timestamp}' if run_timestamp else ''
        path = f'{case_name}_{name}{suffix}.png'
        return os.path.join(output_dir, path) if output_dir else path

    return _save_path


def _prepare_plot_context(mesh, bed_history, bbox, resolution, history, simulation_time):
    # DEM 网格来自真实数据，绘图坐标直接使用 DEM 对应的规则网格。
    nx = int((bbox['xmax'] - bbox['xmin']) / resolution)
    ny = int((bbox['ymax'] - bbox['ymin']) / resolution)
    xc = np.linspace(bbox['xmin'] + resolution / 2, bbox['xmax'] - resolution / 2, nx)
    yc = np.linspace(bbox['ymin'] + resolution / 2, bbox['ymax'] - resolution / 2, ny)
    X, Y = np.meshgrid(xc, yc)
    n_t = len(bed_history)
    time_unit = 'h' if simulation_time > 3600 else 's'
    time_scale = 3600.0 if simulation_time > 3600 else 1.0
    output_times = history.get('output_times') or np.linspace(0.0, simulation_time, n_t).tolist()
    if len(output_times) != n_t:
        output_times = np.linspace(0.0, simulation_time, n_t).tolist()
    time_ids = np.linspace(0, n_t - 1, 6, dtype=int)
    active_mask = getattr(mesh, 'active_cell_mask_2d', None)
    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(ny, nx)

    return {
        'nx': nx,
        'ny': ny,
        'xc': xc,
        'yc': yc,
        'X': X,
        'Y': Y,
        'time_unit': time_unit,
        'time_scale': time_scale,
        'output_times': output_times,
        'time_ids': time_ids,
        'active_mask': active_mask,
    }


def _mask_bed(zb_2d, active_mask):
    if active_mask is None:
        return zb_2d
    return np.where(active_mask, zb_2d, np.nan)


def plot_dem_overview(
    bed_grid,
    active_mask,
    bbox=None,
    save_path=None,
    title='Initial DEM and Active Mask',
):
    """绘制初始 DEM、河道有效单元 mask 和 active 区床面。

    bbox 可以传真实坐标范围 dict，也可以直接传 resolution 标量。
    不传 save_path 时适合在 notebook 中直接显示。
    """
    if not HAS_MATPLOTLIB:
        print("未安装 matplotlib，跳过 DEM 绘图。")
        return

    bed = np.asarray(bed_grid, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    if bed.shape != active.shape:
        raise ValueError("bed_grid 和 active_mask 形状必须一致。")

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    extent = _dem_extent(bed.shape, bbox)
    active_bed = _mask_bed(bed, active)
    active_count = int(np.count_nonzero(active))
    total_count = int(active.size)
    active_ratio = active_count / max(total_count, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(bed, origin='lower', extent=extent, cmap='terrain')
    axes[0].set_title('DEM bed elevation')
    axes[0].set_xlabel('x(m)')
    axes[0].set_ylabel('y(m)')
    plt.colorbar(im0, ax=axes[0], label='zb(m)')

    im1 = axes[1].imshow(active.astype(float), origin='lower', extent=extent, cmap='gray_r', vmin=0, vmax=1)
    axes[1].set_title(f'Active mask: {active_count}/{total_count} ({active_ratio:.1%})')
    axes[1].set_xlabel('x(m)')
    axes[1].set_ylabel('y(m)')
    plt.colorbar(im1, ax=axes[1], label='active')

    im2 = axes[2].imshow(active_bed, origin='lower', extent=extent, cmap='terrain')
    axes[2].set_title('DEM inside active mask')
    axes[2].set_xlabel('x(m)')
    axes[2].set_ylabel('y(m)')
    plt.colorbar(im2, ax=axes[2], label='zb(m)')

    for ax in axes:
        ax.set_aspect('equal')
        ax.grid(alpha=0.15)

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'✓ {save_path}')
        return None

    try:
        from IPython.display import Image, display
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        display(Image(data=buffer.getvalue()))
        plt.close(fig)
    except ImportError:
        plt.show()
    return fig


def _dem_extent(shape, bbox):
    ny, nx = shape
    if bbox is None:
        return [0.0, float(nx), 0.0, float(ny)]
    if isinstance(bbox, dict):
        return [bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']]
    resolution = float(bbox)
    return [0.0, float(nx) * resolution, 0.0, float(ny) * resolution]


def plot_bed_evolution(bed_history, plot_context, save_path):
    nx = plot_context['nx']
    ny = plot_context['ny']
    X = plot_context['X']
    Y = plot_context['Y']
    time_unit = plot_context['time_unit']
    time_scale = plot_context['time_scale']
    output_times = plot_context['output_times']
    time_ids = plot_context['time_ids']
    active_mask = plot_context.get('active_mask')
    masked_history = [
        _mask_bed(np.asarray(zb).reshape(ny, nx), active_mask)
        for zb in bed_history
    ]
    zmin_global = min(float(np.nanmin(zb)) for zb in masked_history)
    zmax_global = max(float(np.nanmax(zb)) for zb in masked_history)
    zmin_plot = min(0.0, zmin_global)
    zmax_plot = max(zmax_global + 0.01, 0.1)
    levels = np.linspace(zmin_plot, zmax_plot, 25)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, tid in zip(axes.flatten(), time_ids):
        zb = masked_history[tid]
        t_v = output_times[tid] / time_scale
        im = ax.contourf(X, Y, zb, levels=levels, cmap='terrain', extend='both')
        ax.contour(X, Y, zb, levels=5, colors='k', linewidths=0.4)
        ax.set_title(f't={t_v:.1f}{time_unit} max={np.nanmax(zb):.3f}m')
        ax.set_aspect('equal')
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        plt.colorbar(im, ax=ax)
    plt.suptitle('Bed Evolution', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n✓ {save_path}')


def plot_bed_change_evolution(bed_history, plot_context, save_path):
    nx = plot_context['nx']
    ny = plot_context['ny']
    X = plot_context['X']
    Y = plot_context['Y']
    time_unit = plot_context['time_unit']
    time_scale = plot_context['time_scale']
    output_times = plot_context['output_times']
    time_ids = plot_context['time_ids']
    active_mask = plot_context.get('active_mask')

    z0 = _mask_bed(np.asarray(bed_history[0]).reshape(ny, nx), active_mask)
    dz_history = [
        _mask_bed(np.asarray(zb).reshape(ny, nx), active_mask) - z0
        for zb in bed_history
    ]
    dz_abs = max(float(np.nanmax(np.abs(dz))) for dz in dz_history)
    dz_abs = max(dz_abs, 1.0e-9)
    levels = np.linspace(-dz_abs, dz_abs, 25)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, tid in zip(axes.flatten(), time_ids):
        dz = dz_history[tid]
        t_v = output_times[tid] / time_scale
        im = ax.contourf(X, Y, dz, levels=levels, cmap='RdBu_r', extend='both')
        ax.contour(X, Y, dz, levels=5, colors='k', linewidths=0.3, alpha=0.5)
        ax.set_title(
            f't={t_v:.1f}{time_unit} '
            f'Δzb=[{np.nanmin(dz):+.2e}, {np.nanmax(dz):+.2e}]m'
        )
        ax.set_aspect('equal')
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        plt.colorbar(im, ax=ax, label='Δzb(m)')
    plt.suptitle('Bed Change Relative to Initial DEM', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ {save_path}')


def plot_bed_profiles(bed_history, plot_context, save_path):
    nx = plot_context['nx']
    ny = plot_context['ny']
    xc = plot_context['xc']
    yc = plot_context['yc']
    time_unit = plot_context['time_unit']
    time_scale = plot_context['time_scale']
    output_times = plot_context['output_times']
    time_ids = plot_context['time_ids']
    active_mask = plot_context.get('active_mask')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # 真实算例没有固定参考断面，默认取 DEM 中心横纵剖面。
    j_mid = len(yc) // 2
    i_mid = len(xc) // 2
    colors = plt.cm.plasma(np.linspace(0, 1, len(time_ids)))
    for c, tid in zip(colors, time_ids):
        zb = _mask_bed(bed_history[tid].reshape(ny, nx), active_mask)
        t_v = output_times[tid] / time_scale
        axes[0].plot(xc, zb[j_mid, :], color=c, lw=2, label=f't={t_v:.1f}{time_unit}')
        axes[1].plot(yc, zb[:, i_mid], color=c, lw=2, label=f't={t_v:.1f}{time_unit}')
    for ax, xl, lb in zip(
        axes,
        [(float(xc[0]), float(xc[-1])), (float(yc[0]), float(yc[-1]))],
        [f'Center row y={yc[j_mid]:.1f}m', f'Center column x={xc[i_mid]:.1f}m'],
    ):
        ax.set_xlim(xl)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylabel('zb(m)')
        ax.set_title(lb)
    axes[0].set_xlabel('x(m)')
    axes[1].set_xlabel('y(m)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ {save_path}')


def plot_training_history(history, plot_context, simulation_time, save_path):
    time_unit = plot_context['time_unit']
    time_scale = plot_context['time_scale']

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))

    def pl(ax, d, lb, c, ls='-'):
        if d:
            ax.semilogy(d, color=c, lw=1.5, ls=ls, label=lb)

    pl(axes[0, 0], history.get('flow_loss', []), 'Flow', 'b')
    pl(axes[0, 0], history.get('continuity', []), 'Cont', 'c', '--')
    pl(axes[0, 0], history.get('flow_boundary_loss', []), 'Boundary raw', 'gray', ':')
    axes[0, 0].set_title('Flow Loss')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    pl(axes[0, 1], history.get('momentum_x', []), 'Mom-x', 'r')
    pl(axes[0, 1], history.get('momentum_y', []), 'Mom-y', 'm')
    axes[0, 1].set_title('Momentum')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    pl(axes[1, 0], history.get('sediment_loss', []), 'Sed', 'g')
    pl(axes[1, 0], history.get('joint_loss', []), 'Joint', 'black', '--')
    pl(axes[1, 0], history.get('transport_loss', []), 'C PDE', 'teal', ':')
    pl(axes[1, 0], history.get('capacity_loss', []), 'Capacity', 'orange', '--')
    pl(axes[1, 0], history.get('bed_change_loss', []), 'Bed Δzb', 'red', '-.')
    pl(axes[1, 0], history.get('initial_sediment_loss', []), 'Initial C', 'purple', '-.')
    pl(axes[1, 0], history.get('inlet_sediment_loss', []), 'Inlet C', 'brown', '-.')
    axes[1, 0].set_title('Sediment Loss')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    if history.get('C_min'):
        axes[1, 1].plot(history['C_min'], 'b-', lw=1.5, label='C min')
        axes[1, 1].plot(history['C_max'], 'r-', lw=1.5, label='C max')
        if history.get('Ceq_mean'):
            axes[1, 1].plot(history['Ceq_mean'], 'k--', lw=1.5, label='C capacity mean')
        axes[1, 1].set_title('Sediment Concentration')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('C')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(alpha=0.3)

    if history.get('dzb_min'):
        axes[2, 0].plot(history['dzb_min'], 'r-', lw=1.5, label='Δzb min')
        axes[2, 0].plot(history['dzb_max'], 'g-', lw=1.5, label='Δzb max')
        axes[2, 0].axhline(0.0, color='k', lw=0.8, alpha=0.5)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Δzb (m)')
        axes[2, 0].set_title('Predicted Bed Change')
        axes[2, 0].legend(fontsize=8)
        axes[2, 0].grid(alpha=0.3)

    if history.get('zb_max'):
        ta_source = history.get('time')
        if ta_source and len(ta_source) == len(history['zb_max']):
            ta = np.asarray(ta_source, dtype=np.float32) / time_scale
        else:
            ta = np.linspace(0, simulation_time / time_scale, len(history['zb_max']))
        axes[2, 1].plot(ta, history['zb_max'], 'g-', lw=2, label='zb max')
        axes[2, 1].plot(ta, history['zb_min'], 'r-', lw=2, label='zb min')
        axes[2, 1].set_title('Bed Elevation Range')
        axes[2, 1].set_xlabel(f'Time({time_unit})')
        axes[2, 1].set_ylabel('zb(m)')
        axes[2, 1].legend(fontsize=8)
        axes[2, 1].grid(alpha=0.3)

    plt.suptitle('Training History')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ {save_path}')


def plot_stage_loss_breakdown(history, save_path):
    """按训练阶段分别绘制各 loss 分项。"""
    plot_phase_flow_losses(history, save_path('phase1_flow_losses'))
    plot_phase_sediment_losses(history, save_path('phase2_sediment_losses'))
    plot_phase_joint_losses(history, save_path('phase3_joint_losses'))


def plot_phase_flow_losses(history, save_path):
    groups = [
        (
            'Raw flow losses',
            [
                ('flow_loss', 'total flow', 'black', '-'),
                ('continuity', 'continuity', 'c', '-'),
                ('momentum_x', 'momentum x', 'r', '-'),
                ('momentum_y', 'momentum y', 'm', '-'),
                ('flow_boundary_loss', 'boundary raw', 'gray', '--'),
            ],
        ),
        (
            'Weighted flow losses',
            [
                ('weighted_continuity', 'weighted continuity', 'c', '-'),
                ('weighted_momentum_x', 'weighted momentum x', 'r', '-'),
                ('weighted_momentum_y', 'weighted momentum y', 'm', '-'),
                ('weighted_flow_boundary', 'weighted boundary', 'gray', '--'),
            ],
        ),
        (
            'Flow adaptive weights',
            [
                ('weight_continuity', 'w continuity', 'c', '-'),
                ('weight_momentum_x', 'w momentum x', 'r', '-'),
                ('weight_momentum_y', 'w momentum y', 'm', '-'),
                ('flow_boundary_weight', 'w boundary', 'gray', '--'),
            ],
        ),
    ]
    _plot_grouped_history(history, groups, 'Phase 1: Flow Training Losses', save_path)


def plot_phase_sediment_losses(history, save_path):
    groups = [
        (
            'Raw sediment losses',
            [
                ('sediment_loss', 'total sediment', 'black', '-'),
                ('transport_loss', 'transport PDE', 'teal', '-'),
                ('inlet_sediment_loss', 'inlet C', 'brown', '-'),
                ('bed_change_loss', 'Exner bed change', 'red', '-'),
                ('initial_sediment_loss', 'initial C', 'purple', '--'),
                ('bed_initial_loss', 'initial dzb', 'orange', '--'),
                ('capacity_loss', 'capacity diagnostic', 'gray', ':'),
            ],
        ),
        (
            'Weighted sediment losses',
            [
                ('weighted_transport_loss', 'weighted transport', 'teal', '-'),
                ('weighted_capacity_loss', 'weighted capacity', 'orange', '--'),
                ('weighted_inlet_sediment_loss', 'weighted inlet', 'brown', '-'),
                ('weighted_bed_change_loss', 'weighted bed change', 'red', '-'),
            ],
        ),
        (
            'Sediment adaptive weights',
            [
                ('weight_transport', 'w transport', 'teal', '-'),
                ('weight_inlet', 'w inlet', 'brown', '-'),
                ('weight_bed_change', 'w bed change', 'red', '-'),
                ('weight_capacity', 'w capacity', 'gray', ':'),
            ],
        ),
    ]
    _plot_grouped_history(history, groups, 'Phase 2: Sediment Training Losses', save_path)


def plot_phase_joint_losses(history, save_path):
    groups = [
        (
            'Joint total and blocks',
            [
                ('joint_loss', 'total joint', 'black', '-'),
                ('joint_flow_loss', 'joint flow block', 'b', '-'),
                ('joint_sediment_loss', 'joint sediment block', 'g', '-'),
                ('weighted_joint_flow_loss', 'weighted flow block', 'navy', '--'),
                ('weighted_joint_sediment_loss', 'weighted sediment block', 'darkgreen', '--'),
                ('joint_flow_boundary_loss', 'joint boundary raw', 'gray', '--'),
            ],
        ),
        (
            'Joint sediment components',
            [
                ('joint_transport_loss', 'transport PDE', 'teal', '-'),
                ('joint_inlet_sediment_loss', 'inlet C', 'brown', '-'),
                ('joint_bed_change_loss', 'Exner bed change', 'red', '-'),
                ('joint_bed_initial_loss', 'initial dzb', 'orange', '--'),
            ],
        ),
        (
            'Coupling diagnostics',
            [
                ('coupling_bed_error', 'bed update error', 'black', '-'),
                ('coupling_joint_loss', 'joint loss per coupling', 'b', ':'),
            ],
        ),
    ]
    _plot_grouped_history(history, groups, 'Phase 3: Joint/Coupled Training Losses', save_path)


def _plot_grouped_history(history, groups, title, save_path):
    if not any(history.get(key) for _, specs in groups for key, _, _, _ in specs):
        return

    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 3.8 * len(groups)), squeeze=False)
    for ax, (group_title, specs) in zip(axes[:, 0], groups):
        has_values = False
        for key, label, color, linestyle in specs:
            values = _positive_history_values(history.get(key, []))
            if values is None:
                continue
            ax.semilogy(values, color=color, lw=1.5, ls=linestyle, label=label)
            has_values = True
        ax.set_title(group_title)
        ax.set_xlabel('Epoch / iteration')
        ax.set_ylabel('Loss / value')
        ax.grid(alpha=0.3)
        if has_values:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ {save_path}')


def _positive_history_values(values):
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= 0.0] = np.nan
    if np.all(np.isnan(arr)):
        return None
    return arr


def plot_flow_adaptive_weights(history, save_path):
    raw_keys = (
        ('continuity', 'Continuity', 'c'),
        ('momentum_x', 'Momentum-x', 'r'),
        ('momentum_y', 'Momentum-y', 'm'),
        ('flow_boundary_loss', 'Boundary', 'gray'),
    )
    weighted_keys = (
        ('weighted_continuity', 'Continuity weighted', 'c'),
        ('weighted_momentum_x', 'Momentum-x weighted', 'r'),
        ('weighted_momentum_y', 'Momentum-y weighted', 'm'),
        ('weighted_flow_boundary', 'Boundary weighted', 'gray'),
    )
    weight_keys = (
        ('weight_continuity', 'w continuity', 'c'),
        ('weight_momentum_x', 'w momentum-x', 'r'),
        ('weight_momentum_y', 'w momentum-y', 'm'),
        ('flow_boundary_weight', 'w boundary', 'gray'),
    )
    if not any(history.get(key) for key, _, _ in weight_keys):
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for key, label, color in raw_keys:
        values = history.get(key, [])
        if values:
            axes[0].semilogy(values, color=color, lw=1.5, label=label)
    axes[0].set_title('Raw Flow Loss Components')

    for key, label, color in weighted_keys:
        values = history.get(key, [])
        if values:
            axes[1].semilogy(values, color=color, lw=1.5, label=label)
    axes[1].set_title('Weighted Flow Loss Components')

    for key, label, color in weight_keys:
        values = history.get(key, [])
        if values:
            axes[2].semilogy(values, color=color, lw=1.5, label=label)
    axes[2].set_title('Adaptive Flow Weights')

    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ {save_path}')
