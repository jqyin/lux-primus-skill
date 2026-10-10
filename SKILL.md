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
cat runs/<jobid>/summary.txt      # first/3rd/last iteration + exit status (written by the job)
```

`run_smoke.sh` prints the job id, log path and output folder. Queue wait varies from seconds to many hours depending on
load (`sbatch --test-only` shows the estimate); check `squeue -u $USER`. Wait for the job to leave the queue,
then read `summary.txt` (or `hpc_jobs/primus-smoke/summarize.sh <log>` on any log).

**Requirements:** a Slurm account (`sacctmgr -nP show assoc user=$USER format=Account`), and the image
`$SIF` (default `$PRIMUS_ROOT/images/primus-v26.5.sif`; override with `export SIF=...`).
No HF token is needed for the smoke test (Mistral-7B-v0.3 tokenizer is ungated, fetched via proxy).

## Layout

| Path | Purpose |
|---|---|
| `hpc_jobs/primus-smoke/` | **Everything that runs on Lux**, as a VISTA HPC job (see "Running through VISTA"). The hand-run scripts below use the same files. |
| `  setup_lux.sh` | Login node, idempotent: clone Primus `release/v26.5`, pin Megatron (`d3528a2`), build `helpers_cpp` in the image (Python 3.12 ABI). Refuses to touch an existing non-git directory. |
| `  job.lux.slurm` | Launcher: apptainer, torchrun, `bond0` master IP, node-local JIT caches; node count from `-N`. Takes `KEY=VALUE` args, writes `train.log` and `summary.txt` to `$VISTA_OUT`. No `#SBATCH` lines. |
| `  cluster_defaults.json` | Resources (1 node, 8 GPUs, exclusive, `batch`, 30 min) and env overrides such as `SIF`. |
| `  README.md` | The job's description for the agent: args and outputs. |
| `  summarize.sh` | Extracts per-iteration metrics from the LAST rank's log lines. |
| `  build_image.sh`, `build_helpers.sh` | Build the image (compute node only) and Megatron's `helpers_cpp`; used by setup and the job. |
| `  *.yaml` | Experiment configs (top-level `modules:`) go to `$PRIMUS_DIR/examples/megatron/configs/MI355X/`, all others are model configs and go to `$PRIMUS_DIR/primus/configs/models/megatron/`, at job start. Flat, because VISTA uploads only the job folder's top-level files. |
| `env.lux.sh` | Site settings for running by hand. All values overridable via env before sourcing. |
| `scripts/setup.sh`, `scripts/run_smoke.sh` | Hand-run wrappers: source env, run `setup_lux.sh` / submit `job.lux.slurm` with `-A $ACCOUNT` and the same resources. |
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

1. Model: add `hpc_jobs/primus-smoke/<name>.yaml` (`extends: llama_base.yaml` for Llama/Mistral-style decoders; set
   `hidden_size`, `ffn_hidden_size`, `num_layers`, `num_attention_heads`, `num_query_groups`, `kv_channels`,
   `rotary_base`, `norm_epsilon`, tokenizer; copy dims from the HF `config.json`). Reference: `hpc_jobs/primus-smoke/mistral_7B_v0.3.yaml`.
2. Experiment: copy `hpc_jobs/primus-smoke/smoke-mistral_7B-bf16-mock.yaml` (keep its top-level `modules:`), point `model:` at it.
3. Run: `scripts/run_smoke.sh 1` with `CONFIG_NAME=<file>.yaml` (e.g. `scripts/run_smoke.sh 1 CONFIG_NAME=x.yaml`).
4. A `${VAR:default}` knob in a YAML only works if the launcher forwards `VAR` into the container. `job.lux.slurm`
   accepts and forwards `CONFIG_NAME`, `GLOBAL_BATCH_SIZE` and `TRAIN_ITERS` (and sets `PRIMUS_WORKSPACE`); to add a
   knob, add it to the `case` that accepts arguments AND as `APPTAINERENV_<VAR>`, then list it in the job's README.
   An unforwarded variable is **silently ignored** (the default is used).

## Running through VISTA

`hpc_jobs/primus-smoke/` follows VISTA's HPC job format (`docs/hpc-jobs.md` in VISTA), so the agent can run it:

```text
submit_hpc_job(job="primus-smoke", cluster="lux", node_count=1)
submit_hpc_job(job="primus-smoke", cluster="lux", node_count=2, script_args="GLOBAL_BATCH_SIZE=512")
get_hpc_job_status(job_id=<id>, cluster="lux")
get_hpc_job_outputs(job_id=<id>, cluster="lux", files=["summary.txt"])
```

VISTA uploads the folder's top-level files (the YAMLs and `summarize.sh`) to `$RUN_DIR_Lux` and runs `setup_lux.sh` on
the login node before every `sbatch`. A missing image or a failed clone is therefore an error from the submit call,
not a failed job. The Primus clone lives in VISTA's per-job shared folder (`$VISTA_JOB_DIR/Primus`), so only the
first submission clones and builds. `cluster_defaults.json` sets `SIF` to the shared image; if it is unreadable, setup
checks only that the user's own image folder is writable, and the job builds the user's own image (see "Image"). Only the `KEY=VALUE` args in the job's README are accepted.

Read `summary.txt` as in "Verified results": exit 0, 20/20 iterations, loss falling.

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

The smoke test uses the shared image `/lustre/orion/world-shared/stf218/junqi/primus/primus-v26.5.sif` (docker
`rocm/primus:v26.5`; the reference package's `env.lux.sh` has its pinned digest). It is only read. If it is missing or unreadable, the job
uses the user's own copy, `USER_SIF` (under VISTA, `images/` beside the job's uploaded sources in the user's own
`<remote dir>.<user>.jobs/primus-smoke/`; by hand, `$PRIMUS_ROOT/images/`), and if that is missing too, builds it
(`hpc_jobs/primus-smoke/build_image.sh`) on its first compute node before training: ~128 GB RAM, node-local
`APPTAINER_TMPDIR`, ~26 min (a login-node build with tmpdir on Lustre did not finish in an hour). Give such a job at
least 1 hour; with under 45 min left it stops at once with exit 3. The image is copied next to `$USER_SIF` and renamed into
place, so a concurrent reader never sees half an image. Set `PRIMUS_IMAGE=rocm/primus@sha256:<digest>` to pin the source.

## Larger workloads (not covered by the smoke test)

Mistral-Large-123B (4+ nodes, FP8, checkpoint conversion, scaling study) is documented in
`lux-primus/Primus_LM_v26.5_fix_4node_Mistral_large_123B/RUNBOOK.lux.md`. Its scripts hardcode account `ven136` and a
`ven136` `PROJ`: override with `ACCOUNT=<acct> PROJ=<lustre path>` before `source env.lux.sh` there, set `SIF` afterwards, and
submit with `sbatch -A <acct>`. It needs a HuggingFace token with access to the gated
`mistralai/Mistral-Large-Instruct-2411` and ~1.6 TB of Lustre. **That path has not been run in this repo yet.**

## Verify after any change

Run `scripts/run_smoke.sh 1`, then read `runs/<jobid>/summary.txt`; expect exit 0, 20/20 iterations, ~900+ TFLOP/s/GPU.
