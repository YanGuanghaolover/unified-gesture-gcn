# Agent Deployment & Verification Guide (`gesture_gcn`)

这份文档供目标机器上的 AI Agent 或开发者直接按步骤完成环境部署、测试验收与后续开发。

## 1. 项目背景

本项目是面向**静态 + 动态统一骨架手势识别**的轻量 PyTorch 工程框架（Plan A），融合了三篇论文的核心设计：
- **DA-GCN (2025)**：4 层轻量骨干 + ARU（注意力拓扑）+ DRU（差分拓扑）+ 通道/时间双流融合（CTF）。
- **DSTSA-GCN (2025)**：分组可学习静态拓扑（消除局部偏置）+ MS-TCN（多尺度膨胀时间卷积 + Shortcut）。
- **Doodle to Detect / SKETCH (NeurIPS)**：安全窗口归一化 + DRE（Dynamic Range Embedding，动静感知门控）+ Joint/Bone Early Fusion。

## 2. 环境要求

- **Python**：推荐 `3.10` – `3.12`（PyTorch 官方 wheel 支持最完善）
- **依赖**：`torch>=2.1.0`, `pydantic>=2.0.0`, `PyYAML>=6.0`, `pytest>=8.0.0`（详见 `pyproject.toml`）
- **硬件**：CPU 即可跑通单元测试与合成 Demo（数秒级）；若有 NVIDIA GPU 会自动启用 CUDA（`device: "auto"`）。

## 3. Agent 一键部署与验收步骤

在解压后的项目根目录执行：

```bash
# Step 1: 创建并激活虚拟环境（三选一，优先用机器上已有的 Conda / uv / venv）
# 方式 A (venv):
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

# 方式 B (conda):
# conda create -n gesture_gcn python=3.11 -y && conda activate gesture_gcn

# Step 2: 安装 PyTorch（如目标环境已有 GPU 版 PyTorch，可跳过此步直接执行 Step 3）
# CPU 版示例：
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 3: 以开发模式安装本项目及测试依赖
pip install -e ".[dev]"

# Step 4: 运行全部单元测试（验证拓扑、预处理、5 组消融梯度流、端到端训练收敛）
pytest -v

# Step 5: 运行完整的静态+动态手势识别 Demo（含单样本推理延迟测速）
python scripts/run_demo.py --config configs/default.yaml
```

## 4. 预期验收标准

1. `pytest -v`：全部测试用例通过（`test_skeleton_transforms.py`、`test_modules.py`、`test_end_to_end.py`），无 `NaN` 梯度，端到端收敛验证准确率 `>= 85%`。
2. `scripts/run_demo.py`：
   - 正常输出模型参数量（约 `0.5M - 0.8M` 级别）。
   - 12 个 Epoch 内 6 类合成手势（3 静态 + 3 动态）验证集准确率应达到 `95% - 100%`。
   - 正常打印单样本推理延迟（ms）与 FPS。

## 5. 后续扩展入口

- **接入真实数据集**（SHREC'17 / DHG-14/28 / MediaPipe 自采数据）：在 `src/gesture_gcn/data/` 新增 `Dataset`，将原始 `(3, T, N)` 序列送入 `SkeletonPreprocessor(spec)` 即可。
- **消融实验**：直接复制修改 `configs/default.yaml` 中的 `use_aru`、`use_dru`、`use_dre`、`use_mstcn`、`use_bone_fusion`、`num_groups`。
