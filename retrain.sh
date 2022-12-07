#!/bin/bash
docker exec -it final_project_model_1 python3 retraining.py ./experiments/trained_models/bert_tuned_2
docker-compose up --build --no-deps model