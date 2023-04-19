#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/MM-DFN"
DATA_DIR="${WORK_DIR}/data/meld/MELD_features_raw1.pkl"

EXP_NO="mmdfn_base_v1"
DATASET="meld"
echo "${EXP_NO}, ${DATASET}"

OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"
MODEL_DIR="${WORK_DIR}/outputs/meld/mmdfn_base/meld_base_6.pkl"

LOG_PATH="${WORK_DIR}/logs/${DATASET}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi


GCN_LAYERS="64 32 16"  # [8 ,16, 32, 64]
LR="0.0005 0.001"   # [0.0005, 0.001]
L2="0.0001 0.0005"  # [0.0001, 0.0005]
DP="0.4 0.2" # [0.2 0.4]
GAMMA="0.5 1"     # [0.5, 1]
SW="0.5-0.5-1.5"
VALID_RATE="0.0" # [0.0, 0.1]

for gcn_layers in ${GCN_LAYERS[@]}
do
for lr in ${LR[@]}
do
for l2 in ${L2[@]}
do
for dropout in ${DP[@]}
do
for gamma in ${GAMMA[@]}
do
for speaker_weights in ${SW[@]}
do
    python -u ${WORK_DIR}/code/run_train_erc.py \
    --dataset ${DATASET^^} \
    --data_dir ${DATA_DIR} \
    --save_model_dir ${OUT_DIR} \
    --speaker_weights ${speaker_weights} \
    --Deep_GCN_nlayers ${gcn_layers} \
    --valid_rate ${VALID_RATE} \
    --modals 'avl' \
    --lr ${lr} \
    --l2 ${l2} \
    --dropout ${dropout} \
    --gamma ${gamma} \
    --reason_flag \
    >> ${LOG_PATH}/${EXP_NO}.out 2>&1

done
done
done
done
done
done