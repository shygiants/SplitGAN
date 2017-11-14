#!/usr/bin/env bash

TRAINER_PACKAGE_PATH=splitgan
MAIN_TRAINER_MODULE=splitgan.trainer

JOB_DIR="/job-dir/splitgan/revised-v7"
#LOG_DIR="no-avg-pool:/job-dir/no-avg-pool,conv-pool:/job-dir/conv-pool,new-arch:/job-dir/new-arch,joint-conv-pool:/job-dir/joint-conv-pool,joint-conv-pool-gamma:/job-dir/joint-conv-pool-gamma"
#LOG_DIR="/job-dir/splitgan"
LOG_DIR="horse-color:/job-dir/cyclegan/horse-color,fashion-synth:/job-dir/cyclegan/fashion-synth"
DATASET_DIR="/dataset"

if [ -z "$2" ]; then
    GPU_OPTION=""
else
    GPU_OPTION="--gpu $2"
fi

function train() {
    python splitgan/trainer.py \
        ${GPU_OPTION} \
        --skip $1 \
        --eval-only false \
        --random-seed 3 \
        --verbosity DEBUG  \
        --job-dir $JOB_DIR \
        --dataset-dir $DATASET_DIR \
        --paired-dataset edges2shoes \
        --model-name splitgan \
        --num-blocks 6 \
        --train-batch-size 1 \
        --eval-batch-size 10 \
        --train-steps 200000 \
        --num-layers 5 \
        --depth 64 \
        --dense-dim 1024 \
        --alpha1 $2 \
        --alpha2 $3 \
        --beta1 $4 \
        --beta2 $5 \
        --lambda1 $6 \
        --lambda2 $7
}

function hypertune() {
    for i in $(seq -w 0.00001 0.000005 0.000021); do #3
        for j in $(seq -w 0.00001 0.000005 0.000021); do #3
            for k in $(seq -w 0.0001 0.00005 0.00021); do #3
                for l in $(seq -w 0.0001 0.00005 0.00021); do #3
                    for m in $(seq -w 5.0 4.0 15.0); do #3
                        for n in $(seq -w 5.0 4.0 15.0); do #3
                            train \
                                true \
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
        false \
        0.00001 \
        0.000001 \
        0.00002 \
        0.00002 \
        1.0 \
        1.0
elif [ $1 = "hypertune" ]; then
    hypertune
elif [ $1 = "tensorboard" ]; then
    tensorboard --logdir=${LOG_DIR} --host=0.0.0.0
elif [ $1 = "convert" ]; then
    python splitgan/datasets/convert_fashion_synth.py \
        ${GPU_OPTION} \
        --dataset-dir $DATASET_DIR
else
    echo "Usage: run.sh [train|hypertune|tensorboard|convert]"
    exit 1
fi
