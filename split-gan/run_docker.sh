#!/bin/bash

DOCKERFILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPOSITORY="shygiants/split-gan"
PORT=""

# TODO: Check the number of gpus
if [ -z "$2" ]; then
    CONTAINER_NAME="sg-$1"
else
    CONTAINER_NAME="sg-$1-gpu-$2"
fi


function print-running() {
    if [ -z "$2" ]; then
        echo "Running $1..."
    else
        echo "Running $1 on GPU $2..."
    fi
}

if [ $1 = "train" ]; then
    print-running $@
elif [ $1 = "tensorboard" ]; then
    print-running $@
    PORT="-p 6006:6006"
else
    echo "Usage: run_docker.sh [train|tensorboard]"
    exit 1
fi

. ${DOCKERFILE_DIR}/config.sh

# Build docker image
nvidia-docker build -t ${REPOSITORY} ${DOCKERFILE_DIR}

# TODO: check if corresponding container is running
# Remove current running container
nvidia-docker stop ${CONTAINER_NAME}
nvidia-docker rm ${CONTAINER_NAME}

# Run docker container
nvidia-docker run --name ${CONTAINER_NAME} ${PORT} -v ${JOB_DIR}:/job-dir -v ${DATASET_DIR}:/dataset -d ${REPOSITORY} $@