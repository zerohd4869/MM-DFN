#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/MM-DFN"
DATA_DIR="${WORK_DIR}/data/iemocap/IEMOCAP_features.pkl"

EXP_NO="mmdfn_base_v1"
DATASET="iemocap"
echo "${EXP_NO}, ${DATASET}"

OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"
MODEL_DIR="${WORK_DIR}/outputs/iemocap/mmdfn_base/mmdfn_base_6.pkl"

LOG_PATH="${WORK_DIR}/logs/${DATASET}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

GCN_LAYERS="16 32"  # [8 ,16, 32, 64]
LR="0.0001 0.0003" # [0.0001, 0.0003]
L2="0.0001 0.0002" # [0.0001 0.0005]
DP="0.2 0.4" # [0.2 0.4]
GAMMA="0.5 1" # [0.5 1]
SW="3-0-1" # 3-0-1

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
    echo "GCN_LAYERS: ${GCN_LAYERS}, LR: ${LR}, L2: ${L2}, DP: ${DP}, GAMMA: ${GAMMA}, SW: ${SW}"
    python -u ${WORK_DIR}/code/run_train_erc.py \
    --dataset ${DATASET^^} \
    --data_dir ${DATA_DIR} \
    --save_model_dir ${OUT_DIR} \
    --speaker_weights ${speaker_weights} \
    --Deep_GCN_nlayers ${gcn_layers} \
    --valid_rate 0.0 \
    --modals 'avl' \
    --lr ${lr} \
    --l2 ${l2} \
    --dropout ${dropout} \
    --gamma ${gamma} \
    --class_weight \
    --reason_flag \
    >> ${LOG_PATH}/${EXP_NO}.out

done
done
done
done
done
done
