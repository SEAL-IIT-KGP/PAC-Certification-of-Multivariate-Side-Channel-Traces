#!/usr/bin/env bash
# Retrain one suite of the released EstraNet checkpoints with the settings used for the paper.
#
#   bi_suite_8datasets/estranet_support/train_checkpoints.sh SUITE [train_trans.py flags]
#
# SUITE is one checkpoint directory of bi_certification_checkpoints.tar.gz:
#   ascadr, ascadf_desync50, ascadf_desync100, aes_rd_fstd_pool5_lr5e4,
#   ches25_lx_w100, ches25_lx_w200, ches25_lx_w500, ches25_lx_w1000, ches25_lx_w3000, ches25_lx
#
# Checkpoints are written to checkpoints/estranet_retrained/SUITE (CKP_DIR overrides it), next to
# the released ones in checkpoints/estranet. A run resumes from the latest checkpoint in that folder,
# so the same command continues an interrupted run. DATA_PATH overrides the trace file, and extra
# flags are passed on to train_trans.py (a later flag overrides the one set here).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ESTRANET="${ESTRANET_ROOT:-$REPO/external/EstraNet}"
SUITE="${1:?usage: train_checkpoints.sh SUITE [train_trans.py flags]}"
shift

# Settings shared by all suites; the case below sets the per-suite values (see README.md).
dataset=ASCAD lr=2.5e-4 clip=0.25 beta=150 steps=4000000 warmup=1000000 iterations=20000
tches20_ascad="datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets/ASCAD_dataset"
case "$SUITE" in
  ascadr)           data=datasets/raw/ascad_random_key/ASCAD.h5; length=1400; pool=10; beta=50 ;;
  ascadf_desync50)  data=$tches20_ascad/ASCAD_desync50.h5; length=700; pool=5 ;;
  ascadf_desync100) data=$tches20_ascad/ASCAD_desync100.h5; length=700; pool=5 ;;
  aes_rd_fstd_pool5_lr5e4)
    data=datasets/raw/aes_rd/AES_RD_estranet_feature_standardized.h5; length=3500; pool=5
    lr=5e-4 clip=1.0 steps=200000 warmup=2000 iterations=2000 ;;
  ches25_lx_w100 | ches25_lx_w200)   dataset=CHES25; length=${SUITE#ches25_lx_w}; pool=5 ;;
  ches25_lx_w500)                    dataset=CHES25; length=500; pool=10 ;;
  ches25_lx_w1000 | ches25_lx_w3000) dataset=CHES25; length=${SUITE#ches25_lx_w}; pool=20 ;;
  ches25_lx)                         dataset=CHES25; length=7000; pool=20 ;;
  *) echo "unknown suite: $SUITE" >&2; exit 2 ;;
esac

if [ "$dataset" = CHES25 ]; then
  data=datasets/raw/ches_ctf_2025/CHES_Challenge.h5
  # Preprocessing of the CHES-CTF-2025 suites, read by data_utils_ches25.py.
  export CHES25_PREPROCESS=lucianoxu CHES25_LX_DENOISE=1 CHES25_LX_PROFILE_TRACES=450000 CHES25_LX_ATTACK_TRACES=-1
  export CHES25_STANDARDIZE=1 CHES25_STANDARDIZE_FIT_TRACES=45000
fi
DATA="${DATA_PATH:-$REPO/$data}"
CKP="${CKP_DIR:-$REPO/checkpoints/estranet_retrained/$SUITE}"
OUT="$REPO/results/estranet_training/$SUITE"

if [ ! -f "$DATA" ]; then
  echo "trace file not found: $DATA (see paper_results/README.md for how to obtain it)" >&2
  exit 1
fi
[ -f "$ESTRANET/transformer.py" ] || "$REPO/paper_results/fetch_estranet.sh" "$ESTRANET"
grep -q CHES25 "$ESTRANET/train_trans.py" || git -C "$ESTRANET" apply "$HERE/train_trans.patch"
mkdir -p "$CKP" "$(dirname "$OUT")"

gpus=$(python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null || echo 0)
echo "suite=$SUITE dataset=$dataset input_length=$length pool_size=$pool beta_hat_2=$beta learning_rate=$lr"
echo "train_steps=$steps data=$DATA checkpoints=$CKP gpus=$gpus"
[ $# = 0 ] || echo "extra train_trans.py flags: $*"
[ "$gpus" != 0 ] || echo "no GPU visible to TensorFlow: a full run takes 5 to 8 hours on one GPU and is impractical on CPU" >&2

cd "$ESTRANET"
python train_trans.py \
  --use_tpu=False --dataset="$dataset" --data_path="$DATA" \
  --checkpoint_dir="$CKP" --warm_start=True --result_path="$OUT" \
  --learning_rate="$lr" --clip="$clip" --min_lr_ratio=0.004 \
  --train_steps="$steps" --warmup_steps="$warmup" --iterations="$iterations" --save_steps=20000 \
  --input_length="$length" --data_desync=0 --train_batch_size=16 --eval_batch_size=16 \
  --n_layer=2 --d_model=128 --d_head=32 --n_head=8 --d_inner=256 \
  --n_head_softmax=8 --d_head_softmax=16 --dropout=0.05 \
  --conv_kernel_size=3 --n_conv_layer=2 --pool_size="$pool" --d_kernel_map=512 --beta_hat_2="$beta" \
  --model_normalization=preLC --head_initialization=forward --softmax_attn=True \
  --max_eval_batch=100 --do_train=True "$@"
