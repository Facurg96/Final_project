FROM python:3.8.13 as base

ADD requirements.txt .
RUN pip3 install -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/src/

ADD ./ /src/

WORKDIR /src

FROM base as test
#RUN ["pytest", "-v", "/src/tests"]

FROM base as build
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
# ENTRYPOINT ["python", "main.py"]