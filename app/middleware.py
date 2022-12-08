import json
import time
from uuid import uuid4

import redis
import settings

# Connecting to Redis 
db = redis.Redis(
    host=settings.REDIS_IP, 
    port=settings.REDIS_PORT, 
    db=settings.REDIS_DB_ID
)


def model_predict(query_dict):
    
    prediction = None
    
    #Generating unique id for the job
    job_id = str(uuid4())
    #Generating the job
    job_data = dict({
        "id": job_id,
        "query_dict":query_dict
    })

    # Send the job to the model service using Redis
    db.lpush(
    settings.REDIS_QUEUE,
    json.dumps(job_data)
    )

    # Loop until we received the response from our ML model
    while True:
        print("esperando", job_id)
        output = db.get(job_id)
        if output:
            output = json.loads(output.decode("utf-8"))
           
            

        #Deleting the job from Redis after we get the results!
            db.delete(job_id)
            break
        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)
    prediction = output["prediction"]
    score = output["score"]

    return prediction, score
    
