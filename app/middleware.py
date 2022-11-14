import json
import time
from uuid import uuid4

import redis
import settings

# TODO DONE
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host=settings.REDIS_IP, 
    port=settings.REDIS_PORT, 
    db=settings.REDIS_DB_ID
)


def model_predict(dictionary):
    
    prediction = None
    
    # Assign an unique ID for this job and add it to the queue.
    # We need to assing this ID because we must be able to keep track
    # of this particular job across all the services
    # TODO DONE

    job_id = str(uuid4())

    # Create a dict with the job data we will send through Redis having the
    # following shape:
    # {
    #    "id": str,
    #    "image_name": str,
    # }
    # TODO DONE
    job_data = dict({
        "id": job_id,
        "dict": dictionary
    })

    # Send the job to the model service using Redis
    # Hint: Using Redis `lpush()` function should be enough to accomplish this.
    # TODO DONE
    db.lpush(
    settings.REDIS_QUEUE,
    json.dumps(job_data)
    )

    # Loop until we received the response from our ML model
    while True:
        # Attempt to get model predictions using job_id
        # Hint: Investigate how can we get a value using a key from Redis
        # TODO DONE
        output = db.get(job_id)
        #a = json.loads(a.decode("utf-8"))
        if output:
            results = json.loads(output.decode("utf-8"))
            prediction = results
            

        # Don't forget to delete the job from Redis after we get the results!
        # Then exit the loop
        # TODO DONE
            db.delete(job_id)
            break
        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    return prediction
    
