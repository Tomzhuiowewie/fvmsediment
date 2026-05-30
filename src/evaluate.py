# evaluate.py – 训练结果可视化
# 绘制床面演化等高线、中心线/横断面剖面和训练历史曲线。

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


def visualize_results(mesh, bed_history, bbox, resolution, history, simulation_time, case_name='default', output_dir=None):
    if not HAS_MATPLOTLIB:
        print('\n未安装 matplotlib，跳过结果绘图；训练历史和 bed_history 仍会返回。')
        return

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    def _save_path(name):
        path = f'{case_name}_{name}.png'
        return os.path.join(output_dir, path) if output_dir else path

    nx = int((bbox['xmax'] - bbox['xmin']) / resolution)
    ny = int((bbox['ymax'] - bbox['ymin']) / resolution)
    xc = np.linspace(bbox['xmin'] + resolution / 2, bbox['xmax'] - resolution / 2, nx)
    yc = np.linspace(bbox['ymin'] + resolution / 2, bbox['ymax'] - resolution / 2, ny)
    X, Y = np.meshgrid(xc, yc)
    n_t = len(bed_history)
    t_u = 'h' if simulation_time > 3600 else 's'
    t_sc = 3600.0 if simulation_time > 3600 else 1.0
    output_times = history.get('output_times') or np.linspace(0.0, simulation_time, n_t).tolist()
    if len(output_times) != n_t:
        output_times = np.linspace(0.0, simulation_time, n_t).tolist()
    tids = np.linspace(0, n_t - 1, 6, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, tid in zip(axes.flatten(), tids):
        zb = bed_history[tid].reshape(ny, nx)
        t_v = output_times[tid] / t_sc
        lv = np.linspace(min(zb.min() - 0.01, -0.05), max(zb.max() + 0.01, 0.1), 25)
        im = ax.contourf(X, Y, zb, levels=lv, cmap='terrain')
        ax.contour(X, Y, zb, levels=5, colors='k', linewidths=0.4)
        ax.set_title(f't={t_v:.1f}{t_u} max={zb.max():.3f}m')
        ax.set_aspect('equal')
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        plt.colorbar(im, ax=ax)
        ax.plot([500, 700, 700, 500, 500], [400, 400, 600, 600, 400], 'r--', lw=1, alpha=0.5)
    plt.suptitle('床面演化', fontsize=13)
    plt.tight_layout()
    plt.savefig(_save_path('bed'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n✓ {_save_path("bed")}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    j500 = np.argmin(np.abs(yc - 500))
    i600 = np.argmin(np.abs(xc - 600))
    colors = plt.cm.plasma(np.linspace(0, 1, len(tids)))
    for c, tid in zip(colors, tids):
        zb = bed_history[tid].reshape(ny, nx)
        t_v = output_times[tid] / t_sc
        axes[0].plot(xc, zb[j500, :], color=c, lw=2, label=f't={t_v:.1f}{t_u}')
        axes[1].plot(yc, zb[:, i600], color=c, lw=2, label=f't={t_v:.1f}{t_u}')
    for ax, xl, lb in zip(
        axes,
        [(300, 800), (300, 700)],
        ['y=500m 中心线 (对照论文Fig.7/12)', 'x=600m 横断面'],
    ):
        ax.set_xlim(xl)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylabel('zb(m)')
        ax.set_title(lb)
    axes[0].set_xlabel('x(m)')
    axes[1].set_xlabel('y(m)')
    plt.tight_layout()
    plt.savefig(_save_path('profiles'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ {_save_path("profiles")}')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    def pl(ax, d, lb, c, ls='-'):
        if d:
            ax.semilogy(d, color=c, lw=1.5, ls=ls, label=lb)
    pl(axes[0, 0], history.get('flow_loss', []), 'Flow', 'b')
    pl(axes[0, 0], history.get('continuity', []), 'Cont', 'c', '--')
    axes[0, 0].set_title('Flow Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    pl(axes[0, 1], history.get('momentum_x', []), 'Mom-x', 'r')
    pl(axes[0, 1], history.get('momentum_y', []), 'Mom-y', 'm')
    axes[0, 1].set_title('Momentum')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    pl(axes[1, 0], history.get('sediment_loss', []), 'Sed', 'g')
    pl(axes[1, 0], history.get('transport_loss', []), 'C PDE', 'teal', ':')
    axes[1, 0].set_title('Sediment Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    if history.get('zb_max'):
        ta_source = history.get('time')
        if ta_source and len(ta_source) == len(history['zb_max']):
            ta = np.asarray(ta_source, dtype=np.float32) / t_sc
        else:
            ta = np.linspace(0, simulation_time / t_sc, len(history['zb_max']))
        axes[1, 1].plot(ta, history['zb_max'], 'g-', lw=2, label='max')
        axes[1, 1].plot(ta, history['zb_min'], 'r-', lw=2, label='min')
        axes[1, 1].set_xlabel(f'Time({t_u})')
        axes[1, 1].set_ylabel('zb(m)')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    plt.suptitle('训练历史')
    plt.tight_layout()
    plt.savefig(_save_path('losses'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ {_save_path("losses")}')
    dz = bed_history[-1].max() - bed_history[0].max()
    print(f'\n  初始峰值: {bed_history[0].max():.4f}m  最终峰值: {bed_history[-1].max():.4f}m')
    print(f'  峰值变化: {dz:+.4f}m ({dz / max(bed_history[0].max(), EPS_SAFE) * 100:.1f}%)')
