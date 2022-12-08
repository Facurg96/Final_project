#!/bin/bash
docker exec -it finalproject_model_1 python3 retraining.py ./experiments/trained_models/bert_tuned
docker-compose up --build --no-deps model