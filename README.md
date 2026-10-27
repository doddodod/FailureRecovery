# Foresight 基础可运行骨架

这是论文 *Foresight: Failure Detection for Long-Horizon Robotic Manipulation
with Action-Conditioned World Model Latents* 的最小工程骨架。它只验证模块边界、
张量数据流、训练与评估流程，不复现论文实验，也不声称 mock 指标有实验意义。

## 文件树

```text
configs/default.yaml
data/{trajectory_dataset.py,mock_data.py}
models/{encoder.py,predictor.py,detector.py,foresight.py}
training/train.py
evaluation/{eval.py,metrics.py}
scripts/{run_sanity.sh,train_mock.sh}
tests/test_sanity.py
config.py
requirements.txt
```

## 核心数据流

1. `Trajectory` 统一保存 `observations [T,O]`、`actions [T,A]`、
   `trajectory_label`（失败 1，成功 0）和可选 `step_label [T]`。
2. `SlidingWindowTrajectoryDataset` 输出
   `observations [S,C,O]` 与 `actions [S,L,A]`。其中 `C` 是观测上下文长度，
   `L` 是动作块长度，`S` 是检测器序列长度。
3. `ObservationEncoder` 只读取观测上下文，产生
   `z^h [B,S,Z]`，绝不读动作。
4. `ActionConditionedPredictor` 读取 `z^h` 和动作块，产生
   `z^p [B,S,Z]`。
5. `FailureDetector` 默认读取 `z^p`；将
   `detector.input_path=hidden` 即可走 `z^h` 消融路径。输出逐步 logits
   `[B,S]`。
6. 损失可选择逐步标签或轨迹标签。评估先合并重叠窗口的逐步概率，再输出
   每轨迹的逐步分数、聚合预测、混淆矩阵、每类准确率和 Balanced Accuracy。

## 运行

```bash
cd /inspurfs/group/hexm/embo/foresight_skeleton
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_sanity.sh
bash scripts/train_mock.sh
```

所有超参位于 `configs/default.yaml`，可重复使用点路径覆盖：

```bash
python -m training.train \
  --config configs/default.yaml \
  --set detector.input_path=hidden \
  --set detector.backbone=mlp \
  --set training.target_level=trajectory
```

可切换的检测器包括 `mlp`、`lstm` 和 `transformer`。可通过
`training.pos_weight` 配置 BCE 正类权重，通过
`training.balanced_sampler=true` 启用窗口级均衡采样。

## 当前假设与 TODO

- 输入暂定为扁平连续向量；图像、多视角和真实 world-model 编码器留给后续接入。
- `observation_context_length` 明确定义为 `k+1`；动作记号
  `a_{t:t+H}` 是否包含 `H+1` 个动作需要对照论文实现，因此直接配置
  `action_chunk_length`。
- 无逐步标签时，逐步损失使用轨迹标签广播作为弱监督；也可切换为轨迹级损失。
- 轨迹分数默认取逐步故障概率最大值；`mean`/`last` 可配置，论文的精确聚合方法
  待确认。
- early-detection weighting 目前只有 hook，尚未实现论文特定权重。
- balanced sampler 当前按滑动窗口的轨迹标签均衡；真实数据的采样单位和划分策略
  待确认。
- mock 数据仅用于 forward/backward、路径切换、checkpoint 和评估管线测试。
