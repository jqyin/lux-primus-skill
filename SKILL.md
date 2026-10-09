---
name: primus-on-lux
description: Run AMD Primus (Megatron backend) LLM training on OLCF Lux (MI355X, Slurm + apptainer). Use when asked to set up, smoke-test, scale, or debug Primus/Megatron training jobs on Lux, add a new model config, or interpret TFLOP/s/GPU results.
---

# Primus on OLCF Lux

Lux: `login1.lux.olcf.ornl.gov`, nodes `lux001..504`, 8x MI355X (gfx950, 288 GB) per node, 128 CPU cores,
8x ionic 400G RoCE NICs, Slurm partition `batch`, container runtime **apptainer** (no podman/docker),
internet **only via proxy** (`proxy.ccs.ornl.gov:3128`). Bulk data goes on Lustre (`/lustre/orion/<proj>/...`);
`$HOME` is small NFS.

Everything here was verified on Lux (see "Verified results"). Read "Gotchas" before changing anything.

## Quickstart

```bash
cd <this repo>
source env.lux.sh                  # detects ACCOUNT; sets PRIMUS_DIR, SIF, proxy, PRIVATE_IFACE
scripts/setup.sh                   # once: clone Primus release/v26.5, pin Megatron, build helpers_cpp in-image
scripts/run_smoke.sh 1             # 1 node, Mistral-7B-v0.3, mock data, 20 iters (~3 min incl. startup)
scripts/run_smoke.sh 2 GLOBAL_BATCH_SIZE=512     # 2 nodes, SAME per-GPU batch (gbs = 256 x nodes)
scripts/summarize.sh runs/smoke-N2_<jobid>.log   # first/3rd/last iteration + exit status
```

`run_smoke.sh` prints the job id and log path. Queue wait varies from seconds to many hours depending on
load (`sbatch --test-only` shows the estimate); check `squeue -u $USER`. Wait for the job to leave the queue,
then summarize.

**Requirements:** a Slurm account (`sacctmgr -nP show assoc user=$USER format=Account`), and the image
`$SIF` (default `$PRIMUS_ROOT/images/primus-v26.5.sif`; override with `export SIF=...`).
No HF token is needed for the smoke test (Mistral-7B-v0.3 tokenizer is ungated, fetched via proxy).

## Layout

| Path | Purpose |
|---|---|
| `env.lux.sh` | The only site-specific file. All values overridable via env before sourcing. |
| `scripts/setup.sh` | Idempotent clone + Megatron pin (`d3528a2`) + `helpers_cpp` build (must run in the container: Python 3.12 ABI). Refuses to touch an existing non-git directory. |
| `scripts/submit_primus_smoke.sbatch` | Self-contained launcher: apptainer, torchrun, `bond0` master IP, node-local JIT caches. Node count comes from `-N`. |
| `scripts/run_smoke.sh` | Wrapper: sources env, applies `KEY=VAL` overrides, submits with `-A $ACCOUNT`. |
| `scripts/summarize.sh` | Extracts per-iteration metrics from the LAST rank's log lines. |
| `configs/*.yaml` | Experiment configs, copied into `$PRIMUS_DIR/examples/megatron/configs/MI355X/` at submit. |
| `models/*.yaml` | Model configs, copied into `$PRIMUS_DIR/primus/configs/models/megatron/` at submit. |
| `lux-primus/` | Reference package for Mistral-Large-123B (image, runbook, scripts). Not required for the smoke test. |
| `runs/` | Job logs (git-ignored). |

## Verified results (Mistral-7B-v0.3, BF16, mock data, seq 2048, mbs 4, Turbo off)

| Nodes | gbs | microbatches/rank | steady TFLOP/s/GPU | step |
|---|---|---|---|---|
| 1 | 256 | 8 | ~944 | 3.25 s |
| 2 | 256 | 4 | ~686 | — |
| 2 | 512 | 8 | ~808-811 (2 runs) | 3.78 s |

Qwen2.5-3B (mbs 10, gbs 640): 1 node ~804, 2 nodes ~767 TFLOP/s/GPU. Healthy runs: exit 0, 20/20 iterations,
loss falling (mock data is memorized, so loss going near 0 is expected and says nothing about quality),
iteration 1-2 slow (JIT, ignore). **Compare scaling at equal per-GPU batch**: halving it cost ~27% per GPU,
a pure batch-size effect; the true 1->2 node cost at equal batch is ~14%.

## Adding a model or experiment

1. Model: add `models/<name>.yaml` (`extends: llama_base.yaml` for Llama/Mistral-style decoders; set
   `hidden_size`, `ffn_hidden_size`, `num_layers`, `num_attention_heads`, `num_query_groups`, `kv_channels`,
   `rotary_base`, `norm_epsilon`, tokenizer; copy dims from the HF `config.json`). Reference: `models/mistral_7B_v0.3.yaml`.
2. Experiment: copy `configs/smoke-mistral_7B-bf16-mock.yaml`, point `model:` at it.
3. Run: `scripts/run_smoke.sh 1` with `CONFIG_NAME=<file>.yaml` (e.g. `scripts/run_smoke.sh 1 CONFIG_NAME=x.yaml`).
4. A `${VAR:default}` knob in a YAML only works if the launcher forwards `VAR` into the container. The launcher
   currently forwards `GLOBAL_BATCH_SIZE` and `TRAIN_ITERS`; add others as `APPTAINERENV_<VAR>` in the sbatch.
   An unforwarded variable is **silently ignored** (the default is used).

## Gotchas (each cost real time)

- **Primus-Turbo is broken in this image.** Stock Primus configs set `enable_primus_turbo: true`; every rank then fails with
  `Primus-Turbo requires amd-aiter==0.1.14.post1`. Set `enable_primus_turbo`, `use_turbo_attention`,
  `use_turbo_grouped_gemm` to `false` (our configs do).
- **Harmless log noise** (do not chase): apex `fused_weight_gradient_mlp_cuda` JIT failure
  (`cuda_runtime_api.h` not found; only an MoE patch is skipped); `libxalt_init.so` LD_PRELOAD error;
  "NCCL WARN dlvsym loaded successfully for ionic_dv_*" (driver load message); `destroy_process_group` warning at exit.
- **Apptainer needs flags podman didn't:** `--rocm` (else no GPU) and an explicit `--bind /lustre/orion` (else the
  directory is silently empty). Pass env with `APPTAINERENV_*`, never `--env` (it comma-splits values such as
  `NCCL_IB_HCA=ionic_0,ionic_1,...` and RCCL quietly uses one NIC).
- **`--cpus-per-task=128` is mandatory** when one task runs a multi-process launcher: without it Slurm confines all
  ranks to one CPU and throughput drops ~5x (reference measured 261 vs 1273 TFLOP/s).
- **JIT caches must be node-local** (`/tmp/$USER/primus`: Triton, inductor, `TORCH_EXTENSIONS_DIR`); `/tmp` is wiped
  between jobs, so each job recompiles. Small models start in ~1 min; the 123B model needs ~20 min.
- **Master address must be an IP on `bond0`**, not a hostname (can resolve to 127.0.1.1 inside the container).
- **Slurm on Lux requires `-A`, `-N`, `-t`** and rejects the `worker` partition; use `batch`. Ask for the shortest
  `-t` you need; short jobs backfill. `--mem=0` is not needed.
- **Never unpack/build many small files on Lustre** (apptainer image builds: use a compute node with node-local tmp).
- **Reading logs:** per-iteration metrics are printed by the last rank (`rank-<n-1>/<n>`), each line twice. Filters
  on "error"/"exit" match Primus's config dump and give false alarms; match `Training execution failed` instead.
- **Rare unexplained hang (seen once, job 84470, 2 nodes).** Iteration 1 printed, then no progress: all 16 GPUs at 100%,
  Python at ~360% CPU, no IB/NCCL error, and no watchdog kill after 14 min. The identical resubmit passed (20/20, same
  master node). Treat as transient: if a job shows no new iteration for ~5 min after iteration 1 (steady steps take 2-4 s),
  `scancel` it and resubmit; keep `-t` short (the sbatch default is 30 min) so a hang costs little. For large jobs add a
  first-iteration watchdog like the reference `submit_train.sbatch` (`FIRST_ITER_TIMEOUT`, `scancel` on silence).
  If it repeats on the same nodes, intersect node lists across failures and `--exclude` them.
- **GitHub through the proxy resets connections sometimes**; `setup.sh` retries the submodule fetch.
- A pre-existing checkout at some path is not yours to overwrite; use a different `PRIMUS_ROOT`.

## Image

The smoke test uses `primus-v26.5.sif` (docker `rocm/primus:v26.5`, pinned digest in
`lux-primus/.../env.lux.sh`). If it is missing, build it on a **compute node** (needs ~128 GB RAM, node-local
`APPTAINER_TMPDIR`, ~26 min; a login-node build with tmpdir on Lustre did not finish in an hour). See
`lux-primus/Primus_LM_v26.5_fix_4node_Mistral_large_123B/scripts/build_sif.sh` and `submit_build_sif.sbatch`.

## Larger workloads (not covered by the smoke test)

Mistral-Large-123B (4+ nodes, FP8, checkpoint conversion, scaling study) is documented in
`lux-primus/Primus_LM_v26.5_fix_4node_Mistral_large_123B/RUNBOOK.lux.md`. Its scripts hardcode account `ven136` and a
`ven136` `PROJ`: override with `ACCOUNT=<acct> PROJ=<lustre path>` before `source env.lux.sh` there, set `SIF` afterwards, and
submit with `sbatch -A <acct>`. It needs a HuggingFace token with access to the gated
`mistralai/Mistral-Large-Instruct-2411` and ~1.6 TB of Lustre. **That path has not been run in this repo yet.**

## Verify after any change

Run `scripts/run_smoke.sh 1`, then `scripts/summarize.sh`; expect exit 0, 20/20 iterations, ~900+ TFLOP/s/GPU.
