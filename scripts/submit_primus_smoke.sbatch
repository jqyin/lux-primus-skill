#!/bin/bash
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 00:30:00
#SBATCH -J primus-smoke
#SBATCH -o %x_%j.log
# Primus smoke test on OLCF Lux: mock data, no checkpoint, ungated tokenizer.
#   sbatch -A <acct> -N 1 scripts/submit_primus_smoke.sbatch [config.yaml]
# Required env (export before sbatch): SIF, PRIMUS_DIR.  Optional: CONFIG_NAME.
set -u
: "${SIF:?path to primus .sif}" "${PRIMUS_DIR:?path to Primus clone}"
# sbatch runs a spooled copy, so the repo root must be given (or submit from <repo>/runs).
SKILL_DIR=${SKILL_DIR:-$(cd "${SLURM_SUBMIT_DIR:-.}/.." && pwd)}
CONFIG=${1:-${CONFIG_NAME:-smoke-mistral_7B-bf16-mock.yaml}}
# Install this repo's model + experiment configs into the Primus tree (Primus only loads configs from there).
cp -f "$SKILL_DIR"/configs/*.yaml "$PRIMUS_DIR/examples/megatron/configs/MI355X/" || echo "WARN: no configs synced from $SKILL_DIR/configs"
cp -f "$SKILL_DIR"/models/*.yaml "$PRIMUS_DIR/primus/configs/models/megatron/" 2>/dev/null || true
IFACE=${PRIVATE_IFACE:-bond0}
NODE_TMP=/tmp/$USER/primus
export http_proxy=http://proxy.ccs.ornl.gov:3128 https_proxy=http://proxy.ccs.ornl.gov:3128
export no_proxy=localhost,127.0.0.1,.ornl.gov,.ccs.ornl.gov

FIRST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER=$(srun -N1 -n1 -w "$FIRST" bash -c "ip -4 addr show $IFACE | grep -oP '(?<=inet )[0-9.]+'" | tail -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export CONFIG IFACE NODE_TMP SIF PRIMUS_DIR NNODES=$SLURM_JOB_NUM_NODES
echo "job $SLURM_JOB_ID nodes=$NNODES master=$FIRST/$MASTER:$MASTER_PORT config=$CONFIG"

srun -N"$NNODES" -n"$NNODES" --ntasks-per-node=1 --cpus-per-task=128 bash -c '
  mkdir -p $NODE_TMP/{triton,inductor,hf,torch_ext}
  # Pass env via APPTAINERENV_* (NOT --env: it comma-splits values like NCCL_IB_HCA).
  export APPTAINERENV_NCCL_SOCKET_IFNAME=$IFACE APPTAINERENV_GLOO_SOCKET_IFNAME=$IFACE
  export APPTAINERENV_NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
  export APPTAINERENV_NCCL_IB_GID_INDEX=1 APPTAINERENV_USING_AINIC=1 APPTAINERENV_NCCL_DEBUG=WARN
  export APPTAINERENV_RCCL_USE_AMD_SMI_LIB=1 APPTAINERENV_CUDA_DEVICE_MAX_CONNECTIONS=1
  export APPTAINERENV_NVTE_USE_RMSNORM_TRITON=1
  export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export APPTAINERENV_PYTHONPATH=/work/Primus:/work/Primus/third_party/Megatron-LM
  export APPTAINERENV_TRITON_CACHE_DIR=$NODE_TMP/triton APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=$NODE_TMP/inductor
  # Config knobs reach the YAML as ${VAR:default}; a var not forwarded here is silently inert.
  export APPTAINERENV_GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-} APPTAINERENV_TRAIN_ITERS=${TRAIN_ITERS:-}
  [ -z "$GLOBAL_BATCH_SIZE" ] && unset APPTAINERENV_GLOBAL_BATCH_SIZE; [ -z "$TRAIN_ITERS" ] && unset APPTAINERENV_TRAIN_ITERS
  export APPTAINERENV_TORCH_EXTENSIONS_DIR=$NODE_TMP/torch_ext APPTAINERENV_HF_HOME=$NODE_TMP/hf APPTAINERENV_HF_HUB_DISABLE_XET=1
  export APPTAINERENV_http_proxy=$http_proxy APPTAINERENV_https_proxy=$https_proxy APPTAINERENV_no_proxy=$no_proxy
  exec apptainer exec --rocm --pwd /work/Primus \
    --bind $PRIMUS_DIR:/work/Primus --bind /lustre/orion:/lustre/orion --bind $NODE_TMP:$NODE_TMP \
    $SIF bash -c "unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN
      exec torchrun --nnodes $NNODES --nproc_per_node 8 --node_rank $SLURM_NODEID \
        --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
        -m primus.cli.main train pretrain --config examples/megatron/configs/MI355X/$CONFIG"
'
echo "=== exit $? ==="
