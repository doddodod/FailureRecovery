# LIBERO-Long dynamics latent baseline

本仓库包含两部分：
main: 基础实验 - 数据源和dynamics latent基础框架
模型v1: dynamics latent的模型和训练代码

## 实验数据

本实验实际使用的数据是
1. NVIDIA
[`LIBERO-Cosmos-Policy`](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy)
数据集的 [`all_episodes`](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy/tree/main/all_episodes)
2. **文件名包含 `suite=libero_10` 的全部 rollout**。`libero_10` 即
LIBERO-Long；它是文件名中的筛选字段，不是一个名为 `suite=libero_10` 的子目录。

筛选后的数据规模：

```text
10 个 task × 每个 task 50 条 = 500 条 rollout
成功：376 条
失败：124 条
总大小：约 4.7 GiB
```

已经完成的 task 0 pilot 也使用这一数据源，具体使用其中 `task=0` 的 50 条
rollout（38 条成功、12 条失败）。集群上的实际数据目录是：

```text
/inspurfs/group/hexm/embo/foresight_skeleton/horizon_experiment/data_full
```

每个 HDF5 的关键字段是：

```text
primary_images_jpeg  [T]     第三人称相机 JPEG bytes
wrist_images_jpeg    [T]     腕部相机 JPEG bytes
proprio              [T, 9]  机器人状态
actions              [T, 7]  该 rollout 实际记录的未来动作
attrs.success                rollout 成功/失败标签
attrs.task_description       任务文本
```

### 下载
仓库已保存完整文件清单：

```text
manifest_full.txt    # 全部 500 条 LIBERO-Long rollout
manifest_task0.txt   # task 0 的 50 条 rollout
```

下载全量数据：

```bash
python download_manifest.py \
  --manifest manifest_full.txt \
  --output data/libero_10 \
  --workers 1
```

下载器使用 Hugging Face 镜像，支持已有文件跳过、临时 `.part` 文件和失败重试。
镜像容易限流，集群 login 节点建议使用 `--workers 1`。

如果需要重新生成 manifest，可从 Hugging Face tree API 取得文件列表，只保留路径中
同时包含 `suite=libero_10` 和 `.hdf5` 的项目。manifest 每行保存一个仓库相对路径，
例如：

```text
all_episodes/episode_data--suite=libero_10--2025_08_03-18_55_47--task=0--ep=1--success=True--regen_demo.hdf5
```

### HDF5 构建格式

如果从自己的 rollout 构建兼容文件，保持以下 schema：

```python
import h5py
import numpy as np

with h5py.File(output_path, "w") as f:
    f.create_dataset("proprio", data=proprio.astype(np.float32))       # [T, 9]
    f.create_dataset("actions", data=actions.astype(np.float32))       # [T, 7]

    jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
    primary = f.create_dataset("primary_images_jpeg", (len(proprio),), dtype=jpeg_dtype)
    wrist = f.create_dataset("wrist_images_jpeg", (len(proprio),), dtype=jpeg_dtype)
    for t in range(len(proprio)):
        primary[t] = np.frombuffer(primary_jpeg_bytes[t], dtype=np.uint8)
        wrist[t] = np.frombuffer(wrist_jpeg_bytes[t], dtype=np.uint8)

    f.attrs["success"] = bool(success)
    f.attrs["task_description"] = task_description
```

要求图像、proprio 和 action 使用同一时间轴，长度均为 `T`。

## 同一 task instance 的成功/失败与 oracle action

配对时至少使用下面的 key：

```text
(suite, collection_run, task)
```

当前示例属于同一个 task instance group：

```text
suite:          libero_10
collection_run: 2025_08_03-18_55_47
task:           0
description:    put both the alphabet soup and the tomato sauce in the basket
```

示例文件：

```text
成功/reference:
episode_data--suite=libero_10--2025_08_03-18_55_47--task=0--ep=1--success=True--regen_demo.hdf5

失败:
episode_data--suite=libero_10--2025_08_03-18_55_47--task=0--ep=11--success=False--regen_demo.hdf5
```

这里的 oracle action condition 不是第三个独立文件：

- 成功/reference action 来自成功文件的 `/actions`；
- 失败 rollout 的 recorded future action 来自失败文件的 `/actions`；
- 从人工标记的 subtask 边界 `t0` 开始，取 `/actions[t0:t0+delta_t]`。

文件名只能保证同 suite、同采集批次和同 task，不能证明两个 episode 的初始物体位姿
完全相同。严格的 same-instance 实验还需要人工 subtask 边界，并按边界处的图像、
proprio 和物体状态进一步匹配。成功和失败样本应对齐到“最后一个成功 subtask 的
结束状态”，再比较下一 subtask 的成功/失败分支。

## Commit `91c3974`：基础 dynamics latent 训练框架

核心入口是 `train_horizon_failure.py`。当前版本是用于验证数据和训练流程的
proprio baseline，尚未接入 DINO 图像 embedding。

输入：

```text
最近 3 步 proprio context + 未来 delta_t 步 recorded/oracle action
delta_t ∈ {1, 2, 4, 8, 16, 32, 64}
```

模型：

```text
proprio context -> MLP observation encoder
action sequence  -> GRU action encoder
task/horizon     -> learned embeddings
concat + MLP     -> 64-D action-conditioned dynamics latent
```

latent 接两个训练头：

```text
future head   -> 预测 proprio[t + delta_t]
failure head  -> 预测 rollout success/failure
```

训练损失：

```text
SmoothL1(predicted_future_proprio, target_future_proprio)
+ weighted BCE(failure_logit, rollout_label)
```

数据按 rollout 划分 train/validation/test，禁止同一 rollout 的滑动窗口跨 split。
`run_task0_pilot.sbatch` 用 1 张 GPU 跑 task 0；`run_full_4gpu.sbatch` 将三个随机
seed 和 shuffled-action 对照分配到 4 张独立 GPU，不使用 DDP。

运行 task 0 baseline：

```bash
python train_horizon_failure.py \
  --data data/libero_10 \
  --task 0 \
  --seed 0 \
  --epochs 40 \
  --output results/task0_pilot.json
```

这个 commit 的作用是提供可运行的 variable-horizon dynamics latent 基线和
rollout-level evaluation 管线，不代表最终的 DINO trajectory-context 模型。
