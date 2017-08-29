#!/usr/bin/env bash

TRAINER_PACKAGE_PATH=splitgan
MAIN_TRAINER_MODULE=splitgan.trainer

JOB_DIR="/job-dir"
DATASET_DIR="/dataset"

if [ -z "$2" ]; then
    GPU_OPTION=""
else
    GPU_OPTION="--gpu $2"
fi


if [ $1 = "train" ]; then
    python splitgan/trainer.py \
        ${GPU_OPTION} \
        --verbosity DEBUG  \
        --job-dir $JOB_DIR \
        --dataset-dir $DATASET_DIR \
        --paired-dataset edges2shoes \
        --train-batch-size 1 \
        --train-steps 100000 \
        --alpha1 0.00001 \
        --alpha2 0.00001 \
        --beta1 0.00015 \
        --beta2 0.00015 \
        --lambda1 4.0 \
        --lambda2 4.0
elif [ $1 = "tensorboard" ]; then
    tensorboard --logdir=$JOB_DIR --host=0.0.0.0
else
    echo "Usage: run.sh [train|tensorboard]"
    exit 1
fi
