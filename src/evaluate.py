# evaluate.py - 训练结果可视化
# 输出三类图：床面等值图、中心剖面、训练损失历史。

import os

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
    plot_bed_profiles(bed_history, plot_context, save_path('profiles'))
    plot_training_history(history, plot_context, simulation_time, save_path('losses'))

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
