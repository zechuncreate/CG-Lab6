# CG-Lab6


# 3D Mesh Reconstruction via Differentiable Rendering

本项目展示了如何利用 **可微渲染（Differentiable Rendering）** 技术，将一个初始的 3D 球体（IcoSphere）逐步变形为目标物体（如经典的奶牛模型）的形状。

## 🚀 项目亮点

* **从 2D 到 3D**：通过对比多个视角下的 2D 剪影（Silhouettes）来引导 3D 几何体的形变。
* **可微渲染管线**：使用 PyTorch3D 的 `SoftSilhouetteShader` 实现梯度的端到端传导。
* **动态可视化**：支持在训练过程中实时观察 3D 顶点的流动与进化。
* **几何约束优化**：集成了拉普拉斯平滑（Laplacian Smoothing）和边长一致性（Edge Loss）等正则化项，确保生成的模型表面平滑。

## 🛠 环境要求

* **Python**: 3.12+ (推荐在 Google Colab 或 Linux 环境运行)
* **GPU**: NVIDIA GPU + CUDA 12.1 (由于涉及复杂的渲染计算，必须开启 GPU 加速)
* **核心库**:
* `torch` & `torchvision`
* `pytorch3d`
* `fvcore`, `iopath`, `ninja`



## 📦 快速安装 (Colab)

由于 PyTorch3D 的环境依赖较为特殊，建议通过源码编译安装：

```bash
pip install fvcore iopath ninja
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

```

## 📖 核心原理

本实验的核心在于 **Loss 函数的组合**。为了让球体变成牛，我们计算了以下四种损失：

1. **Silhouette Loss**: 渲染出的预测剪影与目标剪影之间的均方误差（MSE）。这是形变的主要动力。
2. **Laplacian Smoothing**: 约束顶点的位移，防止产生尖刺，保持表面平滑。
3. **Edge Loss**: 防止三角形面片被拉伸得过于畸形。
4. **Normal Consistency**: 确保相邻面片的法向量方向一致，提升视觉质感。

## 🖥️ 使用说明

1. **准备数据**：确保当前目录下存在 `cow.obj` 文件。
2. **运行脚本**：直接运行主程序。程序会自动执行以下流程：
* 加载并归一化目标模型。
* 初始化一个 4 级细分的等值球。
* 启动基于 Adam/SGD 的优化循环。


3. **查看结果**：
* 每 20 次迭代，屏幕会刷新当前的 3D 形变进度。
* 所有中间状态的 `.obj` 模型将保存在 `./output_meshes/` 目录中。



## 📊 实验表现

在标准的 T4 GPU 上，约 300 次迭代后，你可以观察到球体已经完全拟合了奶牛的轮廓。

| 初始状态 | 中间进化 | 最终形状 |
| --- | --- | --- |
| 完美球体 | 拓扑拉伸 | 奶牛几何体 |

---

## 📂 文件结构

* `cow.obj`: 目标 3D 模型。
* `main.py`: 核心优化与渲染代码。
* `output_meshes/`: 存放导出的进化过程模型文件。

## 🤝 贡献与反馈

如果你对可微渲染或计算几何感兴趣，欢迎提 issue 或参与讨论！
