# Automated product categorization for e-commerce
Final Project for Anyone AI

Fellows: 
    - Facundo Rodriguez Goren
    - German Hilgert
    - Barbara Craig
    - Julian Londoño

Date: 07/12/22

## Before starting the dockers, please download the file tf_model.h5 from this link: 
https://drive.google.com/file/d/18GLpt06r8wZ1mTy_Xamp4hchRIZfaA6v/view?usp=share_link
## and place it inside the model folder, so that the relative path is something like this: 
model\experiments\trained_models\bert_tuned\tf_model.h5
## Install and run

To run the services using compose:

```bash
$ docker-compose up --build -d
```

To stop the services:

```bash
$ docker-compose down
```

## Overview

A natural language processing (NLP) project to create a multi-label system able to classify e-commerce products automatically. This project will result in an API service that is backed by an NLP model that will classify the textual elements of a product record (description, summary etc).

## Introduction

The automation of tasks in the ecommerce world can be of great help to the market, since it focuses the customer on the purchase and not on the configuration of the search or sale.
In this case, it is required to automate tasks in the process of selling a product, adding a predefined category that corresponds to the product they are selling, acquiring the information of the title and the description of the product.

## Folders

- app: FastApi. Web Service.
        -main.py main file for FastApi work
        -middleware.py to ask for predictions from the model
        -schemmas.py where we defined functions and classes for the main file
        -settings.py settings for setting up the environment
- data: 
        -Data cleaning notebook for the EDA.
        -Scripts. 
        -Text normalizer.
        -Data notebook where we created pipelines and tested the models
        -Some files where we exported the models used in the previous versions
- model: Machine Learning models.
        -Preprocessing: file with the pipelines saved for preprocess the data
        -Trained models: where the working model is located
        -Bert_model.py: the definition of the model used
        -contractions.py contractions for text normalization
        -ml_service.py the file that asks for jobs from que queue from redis, makes the predictions and sends back the prediction
        -utils.py classes for the preprocessing pipelines
- feedback: 
        -Feedback.csv for storing wrong predictions
        -save.csv for storing good predictions or products to be shown
- test: test files.
- root: -Docker-compose and requirements to run it. 
        -Retrain.sh file to make the retraining of the model when we accumulate several
        values from feedback
