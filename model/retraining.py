import json
import yaml

import pickle


# Standard Imports
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

# Transformers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder

# Modeling Evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, classification_report
from IPython.display import display, Markdown

# Pipelines
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

#liraries for NLP
import os
import sys
# I had some issues importing files
# This is how I addressed them:
cwd = os.getcwd()
add = "/".join(cwd.split("/")[:-1])
sys.path.append(add)

#from text_normalizer import normalize_corpus
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import Word2Vec
from scipy.sparse import hstack

# Tensorflow
import tensorflow as tf
from tensorflow.keras import activations, optimizers, losses
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from bert_model import *
import sys


# Here are some constants used for creating and training the model:
SAVE_ROUTE = sys.argv[1]

NUM_LABELS = 263
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 20
BATCH_SIZE = 8

modelo = "./experiments/trained_models/bert_tuned"
modelito = TFDistilBertForSequenceClassification.from_pretrained(modelo)

# Setting up some parameters for compiling the model:
optimizer = optimizers.Adam(learning_rate=3e-5)
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
modelito.compile(optimizer=optimizer, loss=loss, 
               metrics=['accuracy'],)


# Making only the classifying layer as trainable
for layer in modelito.layers[:2]:
     layer.trainable = False

print(modelito.summary())


FEEDBACK_CSV = "./feedback/feedback.csv"
PREPROCESSING = "./experiments/preprocessing/text_pipeline.sav"
CATEGORIES = "./experiments/categories_encoded.json"
LABELS = "./experiments/preprocessing/label_pipeline.sav"

# Reading the feedback csv, loading the preprocessing pipeline and so on:
fd = pd.read_csv(FEEDBACK_CSV)
preprocessing = pickle.load(open(PREPROCESSING,"rb"))
labels = BertPredict.get_labels(LABELS)
name_to_label = {(yaml.safe_load(labels.classes_[i])[-1]["name"] if isinstance(yaml.safe_load(labels.classes_[i])[-1], dict) else labels.classes_[i]):labels.classes_[i] for i in range(len(labels.classes_))}

X_retrain = fd.drop(columns="category")
X_retrain = preprocessing.transform(X_retrain)["description"].to_list()


y_retrain = fd["category"].replace(name_to_label)
y_retrain = list(labels.transform(y_retrain))


tkzr = DistilBertTokenizer.from_pretrained(MODEL_NAME)
encodings = BertPredict.construct_encodings(X_retrain, tkzr, max_len=MAX_LEN)
tfdataset = BertPredict.construct_tfdataset(encodings, y_retrain)
tfdataset = tfdataset.batch(BertPredict.BATCH_SIZE)

history = modelito.fit(tfdataset, batch_size=BATCH_SIZE, epochs = 1)

#Saving the new model:
try:
    print("Saving the new model in: ", SAVE_ROUTE)
    modelito.save_pretrained(SAVE_ROUTE)
    print("Successfully saved!!!")
except:
    print("Invalid file path for saving the new model...")