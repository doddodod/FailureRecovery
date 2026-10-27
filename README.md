# LIBERO-Long same-task prefix latent experiment

This directory contains a small exploratory experiment for the question:
does a longer observed trajectory prefix make successful and failed rollouts
more separable in latent space?

## Data source

The two rollouts come from NVIDIA's
[`LIBERO-Cosmos-Policy`](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy)
dataset, directory
[`all_episodes/`](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy/tree/main/all_episodes).
LIBERO-Long is named `libero_10` in the files.

Both rollouts belong to task 0, “put both the alphabet soup and the tomato
sauce in the basket,” and the same collection run (`2025_08_03-18_55_47`):

- `task0_ep1_success.hdf5`: episode 1, success, 388 steps.
- `task0_ep11_failure.hdf5`: episode 11, failure, 285 steps.

Exact upstream files:

- [Success episode](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy/blob/main/all_episodes/episode_data--suite%3Dlibero_10--2025_08_03-18_55_47--task%3D0--ep%3D1--success%3DTrue--regen_demo.hdf5)
- [Failure episode](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy/blob/main/all_episodes/episode_data--suite%3Dlibero_10--2025_08_03-18_55_47--task%3D0--ep%3D11--success%3DFalse--regen_demo.hdf5)

The upstream dataset is released under CC BY 4.0; retain its attribution when
redistributing these samples.

Each HDF5 contains third-person and wrist JPEG observations, 9-D proprioception,
7-D actions, the task description, and the success label. SHA-256 checksums and
the exact source metadata are recorded in `results/results.json`.

## Experiment

At each timestep the experiment concatenates proprioception and action into a
16-D feature, standardizes it, and fits an unsupervised 8-D PCA latent over the
two trajectories. Each rollout is resampled to 64 normalized progress bins.
Latent separation is measured for prefixes of 1, 2, 4, 8, 16, 32, and 64 bins.

Run:

```bash
python run_prefix_latent_experiment.py \
  --success data/raw/task0_ep1_success.hdf5 \
  --failure data/raw/task0_ep11_failure.hdf5 \
  --output-dir results
```

This is a two-rollout exploratory comparison, not a trained failure detector.
It cannot establish classifier generalization or statistical significance. A
proper follow-up should use all same-task episodes with rollout-level splits
and report ROC-AUC and balanced accuracy versus prefix length.

## Result from this pair

The mean aligned latent L2 distance was 1.56 at one normalized progress bin,
2.72 at 16 bins, 3.55 at 32 bins, and 4.53 at the full 64-bin sequence. The
prefix-mean L2 distance reached 2.53 at 64 bins. The early prefix is not
monotonic, but both reported L2 summaries separate more strongly after the
middle of the rollout. See `results/results.json` for exact values and
`results/prefix_latent_separation.png` for the plot.
