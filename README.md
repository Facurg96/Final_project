# Automated product categorization for e-commerce
Final Project for Anyone AI

Fellows: 
    - Facundo Rodriguez Goren
    - German Hilgert
    - Barbara Craig
    - Julian Londoño

Date: 07/12/22

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
- data: Data cleaning. Scripts. Text normalizer.
- model: Machine Learning models.
- feedback: database files.
- test: test files.
- root: Docker-compose and requirements to run it. Also retrain.sh file to make the retraining of the model when we accumulate several
values from feedback
