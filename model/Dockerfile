FROM python:3.8.13 as base

ADD requirements.txt .
RUN pip3 install -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/src/

ADD ./ /src/

WORKDIR /src



FROM base as build
ENTRYPOINT ["python3", "/src/ml_service.py"]
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN python3 -m nltk.downloader stopwords
