#!/bin/sh
export USE_TF=0 
BASE_LOG_DIR=~/experiments/zero_inference/
MODEL_NAME="Llama-2-70b-chat-hf"
FULL_MODEL_NAME="meta-llama/${MODEL_NAME}"


BSZ=1
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 

