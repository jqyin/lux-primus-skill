#!/usr/bin/env bash
# Site layer for running Primus on OLCF Lux.   source env.lux.sh
# Every value is overridable from the environment before sourcing.
export SKILL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

# Slurm account: first one the user has (Lux rejects jobs without -A).
export ACCOUNT=${ACCOUNT:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep -v '^$' | head -1 | tr -d '|')}
export PARTITION=${PARTITION:-batch}

# Bulk data on Lustre only ($HOME is NFS and small). Per-user scratch under the project.
export LUSTRE_ROOT=${LUSTRE_ROOT:-/lustre/orion}
export PRIMUS_ROOT=${PRIMUS_ROOT:-$LUSTRE_ROOT/${ACCOUNT:-unknown}/scratch/$USER/lux-primus-skill}
export PRIMUS_DIR=${PRIMUS_DIR:-$PRIMUS_ROOT/Primus}      # git clone (bind-mounted over /workspace/Primus)
export SIF=${SIF:-$PRIMUS_ROOT/images/primus-v26.5.sif}   # 8.1 GB apptainer image

# Fabric (verified on Lux: bond0 = TCP bootstrap, 8x ionic RoCE for data).
export PRIVATE_IFACE=${PRIVATE_IFACE:-bond0}

# Lux has NO direct egress. sbatch does not source ~/.bashrc, so export here.
export http_proxy=${http_proxy:-http://proxy.ccs.ornl.gov:3128} https_proxy=${https_proxy:-http://proxy.ccs.ornl.gov:3128}
export no_proxy=${no_proxy:-localhost,127.0.0.1,.ornl.gov,.ccs.ornl.gov}
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy NO_PROXY=$no_proxy

[ -n "$ACCOUNT" ] || echo "env.lux.sh: could not detect a Slurm account; export ACCOUNT=<proj>" >&2
