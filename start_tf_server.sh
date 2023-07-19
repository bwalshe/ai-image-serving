#!/bin/bash

MODEL_LOCATION=$1
if [ ! -d "$MODEL_LOCATION" ]; then
  echo "Invalid model location specified"
  echo "Usage: $0 MODEL_LOCATION"
  exit 1
fi

docker run -t --rm -p 8501:8501 -p 8500:8500 \
  --gpus all \
  -v "${MODEL_LOCATION}:/models/" tensorflow/serving:latest-gpu \
  --model_config_file=/models/models.config