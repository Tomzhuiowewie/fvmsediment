# def plot_training_losses(history, save_path='training_losses.png'):
#     """绘制训练loss曲线"""
#     import matplotlib.pyplot as plt
#
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#
#     # Flow loss
#     axes[0].semilogy(history['flow_loss'], 'b-', linewidth=2)
#     axes[0].set_xlabel('Iteration')
#     axes[0].set_ylabel('Loss (log scale)')
#     axes[0].set_title('Flow Network Loss')
#     axes[0].grid(True, alpha=0.3)
#
#     # Sediment loss
#     axes[1].semilogy(history['sediment_loss'], 'r-', linewidth=2)
#     axes[1].set_xlabel('Iteration')
#     axes[1].set_ylabel('Loss (log scale)')
#     axes[1].set_title('Sediment Network Loss')
#     axes[1].grid(True, alpha=0.3)
#
#     # 河床变化
#     axes[2].plot(history['zb_max'], 'g-', linewidth=2, label='zb max')
#     axes[2].plot(history['zb_min'], 'orange', linewidth=2, label='zb min')
#     axes[2].set_xlabel('Time Step')
#     axes[2].set_ylabel('Bed Elevation (m)')
#     axes[2].set_title('Bed Evolution')
#     axes[2].legend()
#     axes[2].grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     print(f"\n✓ Loss曲线已保存: {save_path}")
#     plt.close()
#
# def visualize_hump_evolution(mesh, bed_history, bbox, resolution, history, T_physical=600):
#
#     nx = int((bbox['xmax'] - bbox['xmin']) / resolution)
#     ny = int((bbox['ymax'] - bbox['ymin']) / resolution)
#
#     x_coords = np.linspace(bbox['xmin'] + resolution / 2, bbox['xmax'] - resolution / 2, nx)
#     y_coords = np.linspace(bbox['ymin'] + resolution / 2, bbox['ymax'] - resolution / 2, ny)
#     X, Y = np.meshgrid(x_coords, y_coords)
#
#     # ===== 图1: 河床演变2D等值线 =====
#     fig1 = plt.figure(figsize=(18, 12))
#     gs = GridSpec(2, 3, figure=fig1)
#
#     n_times = len(bed_history)
#     time_indices = [0, n_times // 4, n_times // 2, 3 * n_times // 4, n_times - 1]
#
#     for idx, t_idx in enumerate(time_indices):
#         ax = fig1.add_subplot(gs[idx // 3, idx % 3])
#         zb = bed_history[t_idx].reshape(ny, nx)
#
#         # 计算物理时间（小时）
#         t_hours = (t_idx / (n_times - 1)) * T_physical / 3600
#
#         im = ax.contourf(X, Y, zb, levels=np.linspace(-0.1, 1.0, 23), cmap='terrain')
#         ax.contour(X, Y, zb, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='k', linewidths=0.5)
#         ax.set_title(f't = {t_hours:.1f} h (zb_max = {zb.max():.3f}m)', fontsize=12)
#         ax.set_xlabel('x (m)')
#         ax.set_ylabel('y (m)')
#         ax.set_aspect('equal')
#         plt.colorbar(im, ax=ax, label='zb (m)')
#
#         # 标记初始沙丘位置
#         ax.plot([500, 700, 700, 500, 500], [400, 400, 600, 600, 400], 'r--', linewidth=1, alpha=0.5)
#
#     # 第6个子图: 中心线剖面 (y=500m，沙丘中心)
#     ax = fig1.add_subplot(gs[1, 2])
#     center_j = ny // 2  # y=500m
#
#     colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
#     for i, t_idx in enumerate(time_indices):
#         zb = bed_history[t_idx].reshape(ny, nx)
#         # 找到y=500m对应的行
#         y_target = 500
#         j_idx = np.argmin(np.abs(y_coords - y_target))
#         t_hours = (t_idx / (n_times - 1)) * T_physical / 3600
#         ax.plot(x_coords, zb[j_idx, :], color=colors[i], label=f't={t_hours:.0f}h', linewidth=2)
#
#     ax.set_xlabel('x (m)')
#     ax.set_ylabel('zb (m)')
#     ax.set_title('Center Line Profile (y=500m)')
#     ax.legend(loc='upper right')
#     ax.grid(True, alpha=0.3)
#     ax.set_xlim(400, 800)
#     ax.set_ylim(-0.1, 1.1)
#     ax.axvline(x=500, color='r', linestyle='--', alpha=0.3, label='Initial hump')
#     ax.axvline(x=700, color='r', linestyle='--', alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig('paper_case_bed_evolution.png', dpi=150, bbox_inches='tight')
#     print("\n✓ 河床演变图已保存: paper_case_bed_evolution.png")
#
#     # ===== 图2: 论文风格对比图 (类似Figure 3) =====
#     fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
#
#     # 左上: 沿x方向的剖面 (y=500m)
#     ax = axes[0, 0]
#     y_target = 500
#     j_idx = np.argmin(np.abs(y_coords - y_target))
#
#     for i, t_idx in enumerate([0, n_times // 2, n_times - 1]):
#         zb = bed_history[t_idx].reshape(ny, nx)
#         t_hours = (t_idx / (n_times - 1)) * T_physical / 3600
#         ax.plot(x_coords, zb[j_idx, :], label=f't={t_hours:.0f}h', linewidth=2)
#
#     ax.set_xlabel('x (m)')
#     ax.set_ylabel('Bed level zb (m)')
#     ax.set_title('Bed Profile along y=500m')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     ax.set_xlim(400, 800)
#
#     # 右上: 沿y方向的剖面 (x=600m)
#     ax = axes[0, 1]
#     x_target = 600
#     i_idx = np.argmin(np.abs(x_coords - x_target))
#
#     for i, t_idx in enumerate([0, n_times // 2, n_times - 1]):
#         zb = bed_history[t_idx].reshape(ny, nx)
#         t_hours = (t_idx / (n_times - 1)) * T_physical / 3600
#         ax.plot(y_coords, zb[:, i_idx], label=f't={t_hours:.0f}h', linewidth=2)
#
#     ax.set_xlabel('y (m)')
#     ax.set_ylabel('Bed level zb (m)')
#     ax.set_title('Bed Profile along x=600m')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     ax.set_xlim(300, 700)
#
#     # 左下: 峰值演变
#     ax = axes[1, 0]
#     if len(history['zb_max']) > 0:
#         # x轴转换为物理时间（小时）
#         time_hours = np.linspace(0, T_physical / 3600, len(history['zb_max']))
#         ax.plot(time_hours, history['zb_max'], 'b-', linewidth=2, label='zb_max')
#         ax.plot(time_hours, history['zb_min'], 'r-', linewidth=2, label='zb_min')
#     ax.set_xlabel('Time (hours)')
#     ax.set_ylabel('Bed Level (m)')
#     ax.set_title('Bed Level Evolution')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     # 右下: 损失曲线
#     ax = axes[1, 1]
#     if len(history['sediment_loss']) > 0:
#         ax.semilogy(history['sediment_loss'], 'g-', alpha=0.7, label='Sediment Loss')
#     ax.set_xlabel('Training Iteration')
#     ax.set_ylabel('Loss')
#     ax.set_title('Training Loss')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig('paper_case_analysis.png', dpi=150, bbox_inches='tight')
#     print("✓ 分析图已保存: paper_case_analysis.png")
#
#     # ===== 图3: 3D视图 =====
#     fig3 = plt.figure(figsize=(16, 6))
#
#     for idx, t_idx in enumerate([0, n_times // 2, n_times - 1]):
#         ax = fig3.add_subplot(1, 3, idx + 1, projection='3d')
#         zb = bed_history[t_idx].reshape(ny, nx)
#         t_hours = (t_idx / (n_times - 1)) * T_physical / 3600
#
#         # 只绘制沙丘区域
#         x_mask = (x_coords >= 400) & (x_coords <= 800)
#         y_mask = (y_coords >= 300) & (y_coords <= 700)
#
#         X_sub = X[np.ix_(y_mask, x_mask)]
#         Y_sub = Y[np.ix_(y_mask, x_mask)]
#         Z_sub = zb[np.ix_(y_mask, x_mask)]
#
#         surf = ax.plot_surface(X_sub, Y_sub, Z_sub, cmap='terrain',
#                                linewidth=0, antialiased=True, alpha=0.9)
#         ax.set_xlabel('x (m)')
#         ax.set_ylabel('y (m)')
#         ax.set_zlabel('zb (m)')
#         ax.set_title(f't = {t_hours:.0f} h')
#         ax.set_zlim(-0.1, 1.1)
#         ax.view_init(elev=30, azim=45)
#
#     plt.tight_layout()
#     plt.savefig('paper_case_3d.png', dpi=150, bbox_inches='tight')
#     print("✓ 3D视图已保存: paper_case_3d.png")
#
#     # ===== 图4: 水深和流速场 =====
#     fig4, axes = plt.subplots(1, 3, figsize=(15, 4))
#
#     # 计算初始水深和流速
#     zb_initial = bed_history[0].reshape(ny, nx)
#     h_initial = 10.0 - zb_initial
#     u_initial = 10.0 / h_initial  # Q/h
#
#     im0 = axes[0].contourf(X, Y, zb_initial, levels=20, cmap='terrain')
#     axes[0].set_title('Initial Bed zb (m)')
#     axes[0].set_aspect('equal')
#     plt.colorbar(im0, ax=axes[0])
#
#     im1 = axes[1].contourf(X, Y, h_initial, levels=20, cmap='Blues')
#     axes[1].set_title('Initial Water Depth h (m)')
#     axes[1].set_aspect('equal')
#     plt.colorbar(im1, ax=axes[1])
#
#     im2 = axes[2].contourf(X, Y, u_initial, levels=20, cmap='Reds')
#     axes[2].set_title('Initial Velocity u (m/s)')
#     axes[2].set_aspect('equal')
#     plt.colorbar(im2, ax=axes[2])
#
#     plt.tight_layout()
#     plt.savefig('paper_case_initial_conditions.png', dpi=150, bbox_inches='tight')
#     print("✓ 初始条件图已保存: paper_case_initial_conditions.png")
#
#     # 打印统计信息
#     print("\n" + "=" * 60)
#     print(" 结果统计")
#     print("=" * 60)
#     print(f"  初始河床峰值: {bed_history[0].max():.4f} m")
#     print(f"  最终河床峰值: {bed_history[-1].max():.4f} m")
#     print(f"  峰值变化: {bed_history[-1].max() - bed_history[0].max():.4f} m")
#     print(f"  相对变化: {(bed_history[-1].max() - bed_history[0].max()) / bed_history[0].max() * 100:.2f}%")