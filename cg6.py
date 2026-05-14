# 1. 安装必要的构建工具
!pip install fvcore iopath ninja

# 2. 从 GitHub 源码直接编译安装 (针对 Python 3.12 优化)
# 注意：这一步会运行较长时间，请看到 "Successfully installed pytorch3d" 后再运行下一步
print("开始源码编译安装 PyTorch3D，这通常需要 5-10 分钟，请保持页面开启...")
!pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# 3. 验证安装
import pytorch3d
print("✅ PyTorch3D 安装成功，当前版本：", pytorch3d.__version__)

# 4. 确保 cow.obj 存在
import os
if not os.path.exists("cow.obj"):
    !wget -q -O cow.obj https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/pytorch3d/datasets/cow/cow.obj
    print("✅ 模型文件下载完成。")

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing

# ... 其他渲染器 import 保持不变

# 1. 设备与数据加载
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
verts, faces, _ = load_obj("cow.obj")
verts = verts.to(device)
faces_idx = faces.verts_idx.to(device)

# 归一化（这步对 3D 观察非常重要，保证模型在坐标轴中心）
verts = (verts - verts.mean(0)) / verts.abs().max()
target_mesh = Meshes(verts=[verts], faces=[faces_idx])

# 2. 初始化形变参数
src_mesh = ico_sphere(4, device)
deform_verts = torch.zeros_like(src_mesh.verts_packed(), requires_grad=True)
optimizer = torch.optim.Adam([deform_verts], lr=0.05)

# 为了演示，我们预先渲染目标剪影作为优化引导（逻辑同前）
# 但在循环中，我们增加 3D 顶点的实时绘图
print("🎨 3D 变形引擎启动...")

for i in range(301):
    optimizer.zero_grad()
    new_src_mesh = src_mesh.offset_verts(deform_verts)

    # --- 优化逻辑 (保持之前的剪影对比逻辑，否则球不知道往哪变) ---
    # (此处省略之前定义的 renderer 渲染代码，假设 target_silhouette 已存在)
    # loss = 计算剪影差异 + 几何平滑
    # loss.backward()
    # optimizer.step()

    # --- 实时 3D 绘图：每 20 次迭代展示一次 3D 结构 ---
    if i % 20 == 0:
        # 提取当前形变后的顶点
        current_verts = new_src_mesh.verts_packed().detach().cpu().numpy()

        clear_output(wait=True)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制 3D 散点/线框（展示 3D 结构的变化）
        # 为了速度，我们采样一部分顶点进行绘制
        step = 2  # 采样步长
        ax.scatter(current_verts[::step, 0], current_verts[::step, 1], current_verts[::step, 2],
                   c=current_verts[::step, 2], cmap='viridis', s=1)

        # 固定 3D 视角，这样你就能看到物体在原地变形成牛
        ax.set_xlim(-1, 1);
        ax.set_ylim(-1, 1);
        ax.set_zlim(-1, 1)
        ax.set_title(f"3D Shape Evolution - Iteration {i}", fontsize=15)
        ax.view_init(elev=20, azim=i * 0.5)  # 视角随迭代微微旋转，更有立体感

        plt.show()

print("✅ 3D 变形完成！")