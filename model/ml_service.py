
import json
import os
import time
import pandas as pd
import numpy as np
import redis
import settings
import pickle
#import model
from bert_model import *

pre = "./experiments/preprocessing/text_pipeline.sav"
modelo = "./experiments/trained_models/bert_tuned"
cats = "./experiments/categories_encoded.json"
model = BertPredict(pre,cats,modelo)

db = redis.Redis(
    host=settings.REDIS_IP, 
    port=settings.REDIS_PORT, 
    db=settings.REDIS_DB_ID
)


def predict(query_dict):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    preds = model(**query_dict)

    # So we access to the class name and the predicted probability with the following lines:
    class_name = preds[0]
    pred_probability = preds[1]

    return class_name, round(pred_probability,4)
    

def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
        # Hint: You should be able to successfully implement the communication
        #       code with Redis making use of functions `brpop()` and `set()`.
        # TODO
        # 1. Taking a new job from Redis
        queue_name, msg = db.brpop(settings.REDIS_QUEUE)
        msg = json.loads(msg) #converts the json object to a python dictionary
        job_id = msg["id"]
        query_dict = msg["query_dict"]

        # 2. Running the ML model on the given data:
        class_name, probability = predict(query_dict)

        # 3. Storing as a Python dictionary:
        results = {
            "prediction":class_name,
            "score":float(probability),
        }

        # 4. Storing the results on Redis Hash Table:
        db.set(job_id, json.dumps(results))
        # Don't forget to sleep for a bit at the end
        time.sleep(settings.SERVER_SLEEP)


      

if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
