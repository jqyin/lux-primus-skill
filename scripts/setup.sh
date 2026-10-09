#!/usr/bin/env bash
# One-time setup (login node, a few minutes). Idempotent.  source env.lux.sh && scripts/setup.sh
# Clones Primus release/v26.5, pins Megatron-LM, builds helpers_cpp inside the image.
set -eu
: "${PRIMUS_DIR:?source env.lux.sh first}" "${SIF:?source env.lux.sh first}"
MEGATRON_PIN=${MEGATRON_PIN:-d3528a21301db2d12e92912b3ec025dc8a2ed4d6}   # what the image's Primus was validated with
mkdir -p "$(dirname "$PRIMUS_DIR")"
[ -s "$SIF" ] || { echo "ERROR: no image at $SIF (see SKILL.md 'Image')"; exit 1; }
if [ -e "$PRIMUS_DIR" ] && ! git -C "$PRIMUS_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: $PRIMUS_DIR exists but is not a usable git clone. Not touching it; set PRIMUS_ROOT/PRIMUS_DIR elsewhere."; exit 1
fi
if [ ! -e "$PRIMUS_DIR" ]; then
  git clone --branch release/v26.5 https://github.com/AMD-AGI/Primus.git "$PRIMUS_DIR"
fi
# GitHub through the proxy resets connections now and then; retry the submodule.
for i in 1 2 3; do git -C "$PRIMUS_DIR" submodule update --init --recursive third_party/Megatron-LM && break || sleep 5; done
MEG=$PRIMUS_DIR/third_party/Megatron-LM
[ "$(git -C "$MEG" rev-parse HEAD)" = "$MEGATRON_PIN" ] || {
  git -C "$MEG" fetch --depth 50 origin "$MEGATRON_PIN" 2>/dev/null || git -C "$MEG" fetch origin
  git -C "$MEG" checkout --detach "$MEGATRON_PIN"; }
# helpers_cpp must be compiled with the container's Python ABI (3.12), so build it in the image.
mkdir -p "$PRIMUS_ROOT/scratch" /tmp/$USER/primus
apptainer exec --rocm --bind "$LUSTRE_ROOT:$LUSTRE_ROOT" --bind "$PRIMUS_DIR:/work/Primus" --pwd /work/Primus/third_party/Megatron-LM/megatron/core/datasets "$SIF" make 2>&1 | tail -2
ls "$MEG"/megatron/core/datasets/helpers_cpp*.so
echo "setup OK: Primus $(git -C "$PRIMUS_DIR" rev-parse --short HEAD), Megatron ${MEGATRON_PIN:0:9}"
