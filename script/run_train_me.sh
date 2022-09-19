#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="./MM-DFN"


EXP_NO="mmdfn_base"
DATASET="meld"
echo "${EXP_NO}, ${DATASET}"

DATA_DIR="${WORK_DIR}/data/${DATASET}/MELD_features_raw1.pkl"
OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"
MODEL_DIR="${WORK_DIR}/outputs/meld/mmdfn_base/meld_base_6.pkl"

LOG_PATH="${WORK_DIR}/logs/${DATASET}"
echo "123"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi


python -u ${WORK_DIR}/code/run_train_erc.py \
--dataset ${DATASET^^} \
--data_dir ${DATA_DIR} \
--save_model_dir ${OUT_DIR} \
--speaker_weights '0.5-0.5-1.5' \
--Deep_GCN_nlayers 32 \
--valid_rate 0.0 \
--modals 'avl' \
--lr 0.001 \
--l2 0.0005 \
--gamma 1 \
--reason_flag \
>> ${LOG_PATH}/${EXP_NO}.out 2>&1
