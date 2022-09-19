#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="./MM-DFN"


EXP_NO="mmdfn_base"
DATASET="iemocap"
echo "${EXP_NO}, ${DATASET}"

DATA_DIR="${WORK_DIR}/data/${DATASET}/IEMOCAP_features.pkl"
OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"
MODEL_DIR="${WORK_DIR}/outputs/iemocap/mmdfn_base/mmdfn_base_6.pkl"


LOG_PATH="${WORK_DIR}/logs/${DATASET}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

python -u ${WORK_DIR}/code/run_train_erc.py \
--dataset ${DATASET^^} \
--data_dir ${DATA_DIR} \
--save_model_dir ${OUT_DIR} \
--speaker_weights '3-0-1' \
--Deep_GCN_nlayers 16 \
--valid_rate 0.0 \
--modals 'avl' \
--lr 0.0003 \
--l2 0.0001 \
--gamma 0.5 \
--class_weight \
--reason_flag \
> ${LOG_PATH}/${EXP_NO}.out
