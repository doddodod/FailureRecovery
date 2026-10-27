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

This version uses the actual modules in `/inspurfs/group/hexm/embo/foresight_skeleton`:
`ObservationEncoder` produces the observation-only hidden latent `z^h`, and
`ActionConditionedPredictor` produces `z^p` from `z^h` plus an action chunk.
There is deliberately no checkpoint and no training. The modules are evaluated
at 10 fixed random initializations (seeds 0-9), so the plot shows mean ± standard
deviation rather than relying on one lucky random projection.

Proprioception and actions are standardized jointly across the pair. The default
repository windows are used (3 observation steps and 4 action steps), and each
valid-window latent sequence is resampled to 64 normalized progress bins.
Separation is measured for prefixes of 1, 2, 4, 8, 16, 32, and 64 bins.

Run:

```bash
python run_repo_latent_timeline.py \
  --success data/raw/task0_ep1_success.hdf5 \
  --failure data/raw/task0_ep11_failure.hdf5 \
  --output-dir results
```

This is a two-rollout geometric comparison, not a trained failure detector.
Random untrained latents have no learned failure semantics. The experiment can
answer whether this particular pair moves farther apart under the repository's
random feature mapping, but it cannot establish classifier generalization or
statistical significance.

## Result from this pair

The result was generated on the login node because this small forward-only MLP
experiment does not require a GPU. For `z^h`, mean aligned L2 rises from
`0.130 ± 0.028` at 1/64 progress to `0.975 ± 0.111` over all 64 bins; its
prefix-centroid L2 rises from `0.130 ± 0.028` to `0.624 ± 0.127`. Thus this pair
does show a strong longer-horizon separation in the observation latent.

For action-conditioned `z^p`, mean aligned L2 is higher at 64 bins than at one
bin (`0.721 ± 0.027` versus `0.411 ± 0.116`), but it is not monotonic. More
importantly, prefix-centroid L2 decreases from `0.411 ± 0.116` to
`0.128 ± 0.034`. The pair therefore does **not** support an unconditional claim
that every action-conditioned prefix representation becomes more separable as
the timeline grows. See `results/results.json` for all per-seed and aggregate
values and `results/prefix_latent_separation.png` for the plot.
