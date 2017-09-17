#!/usr/bin/env bash

TRAINER_PACKAGE_PATH=splitgan
MAIN_TRAINER_MODULE=splitgan.trainer

JOB_DIR="/job-dir"
LOG_DIR="/job-dir"
DATASET_DIR="/dataset"

if [ -z "$2" ]; then
    GPU_OPTION=""
else
    GPU_OPTION="--gpu $2"
fi

function train() {
    python splitgan/trainer.py \
        ${GPU_OPTION} \
        --verbosity DEBUG  \
        --job-dir $JOB_DIR \
        --dataset-dir $DATASET_DIR \
        --paired-dataset edges2shoes \
        --train-batch-size 1 \
        --train-steps 200000 \
        --num-layers 4 \
        --depth 64 \
        --split-rate 2 \
        --alpha1 $1 \
        --alpha2 $2 \
        --beta1 $3 \
        --beta2 $4 \
        --lambda1 $5 \
        --lambda2 $6
}

function hypertune() {
    for i in $(seq -w 0.00001 0.000005 0.000021); do #3
        for j in $(seq -w 0.00001 0.000005 0.000021); do #3
            for k in $(seq -w 0.0001 0.00005 0.00021); do #3
                for l in $(seq -w 0.0001 0.00005 0.00021); do #3
                    for m in $(seq -w 5.0 4.0 15.0); do #3
                        for n in $(seq -w 5.0 4.0 15.0); do #3
                            train \
                                ${i} \
                                ${j} \
                                ${k} \
                                ${l} \
                                ${m} \
                                ${n}
                        done
                    done
                done
            done
        done
    done
}
if [ $1 = "train" ]; then
    train \
        0.00001 \
        0.00001 \
        0.00015 \
        0.00015 \
        10.0 \
        10.0
elif [ $1 = "hypertune" ]; then
    hypertune
elif [ $1 = "tensorboard" ]; then
    tensorboard --logdir=${LOG_DIR} --host=0.0.0.0
else
    echo "Usage: run.sh [train|hypertune|tensorboard]"
    exit 1
fi
