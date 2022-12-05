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

from text_normalizer import normalize_corpus
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

# Required classes:
class DropColumns(TransformerMixin):
    """This Transformer, receives a list of column names
    as a parameter, which are the columns to be dropped."""
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop
    def transform(self, X):
        return X.drop(self.cols_to_drop, axis=1)

    def fit(self, X, y=None):
        return self

class ColumnExtractor(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def transform(self, X):
        return X[self.columns]

    def fit(self, X, y=None):
        return self
class MyOneHotEncoding(TransformerMixin):
    def transform(self, X):
        return pd.DataFrame(self.ohe.transform(X), columns = self.ohe_categories)

    def fit(self, X, y=None):
        self.ohe = OneHotEncoder(sparse=False).fit(X)
        self.ohe_categories = self.ohe.categories_
        return self 
        
class NameDescriptionImputation(TransformerMixin):
    #def __init__(self, to_keep):
    #    self.to_keep = to_keep
    def transform(self, X):
        X["name"] = X["name"].fillna(X["description"])
        X["name"] = X["name"].fillna("UNK")
        X["description"] = X["description"].fillna(X["name"])
        X["description"] = X["description"].fillna("UNK")
        return X

    def fit(self, X_param, y=None):
        return self

class NameDescriptionNormalization(TransformerMixin):
    def transform(self, X):
        new_df = pd.DataFrame({"name":normalize_corpus(list(X["name"])),
                                 "description":normalize_corpus(list(X["description"]))
                                })
        return new_df

    def fit(self, X, y=None):
        return self
        
class BertPredict():
    NUM_LABELS = 263
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LEN = 20
    BATCH_SIZE = 8
    def __init__(self, preprocessing:str, categories:str, model:str):
        self.preprocessing = pickle.load(open(preprocessing,"rb"))
        self.model = TFDistilBertForSequenceClassification.from_pretrained(model)
        with open(categories, 'r') as j:
            self.categories = json.loads(j.read())
        self.predictor = BertPredict.create_predictor(model = self.model,model_name= BertPredict.MODEL_NAME,max_len= BertPredict.MAX_LEN)
    def __call__(self, name, description, types=None, price=None):
        types = np.nan if types==None else types
        price = np.nan if price==None else price

        sku = np.nan; upc = np.nan; shipping = np.nan;
        manufacturer= np.nan; model = np.nan; url = np.nan; image = np.nan
        x_test = pd.DataFrame({
            "sku":pd.Series([sku],dtype="float"),
            "name":pd.Series([name],dtype="str"),
            "type":pd.Series([types],dtype="str"),
            "price":pd.Series([price],dtype="float"),
            "upc":pd.Series([upc],dtype="float"),
            "shipping":pd.Series([shipping],dtype="float"),
            "description":pd.Series([description], dtype="str"),
            "manufacturer":pd.Series([manufacturer], dtype="str"),
            "model":pd.Series([model], dtype="str"),
            "url":pd.Series([url],dtype="str"),
            "image":pd.Series([image], dtype="str")
        })
        x_test_transformed = self.preprocessing.transform(x_test)["description"]
        preds = self.predictor(x_test_transformed) 
        #name_pred = self.categories[str(pred[0])]["name"]
        #prob = self.model.predict_proba(x_test_transformed)[0][pred[0]]
        preds = list(map(lambda x: (self.categories[str(x[0])]["name"], x[1]), preds))
        #return (name_pred, prob)
        return preds[0]

    @staticmethod
    def construct_encodings(x, tkzr, max_len, trucation=True, padding=True):
        return tkzr(x, max_length=max_len, truncation=trucation, padding=padding)

    @staticmethod
    def construct_tfdataset(encodings, y=None):
        if y:
            return tf.data.Dataset.from_tensor_slices((dict(encodings),y))
        else:
            # this case is used when making predictions on unseen samples after training
            return tf.data.Dataset.from_tensor_slices(dict(encodings))

    @staticmethod
    def create_predictor(model, model_name, max_len):
        tkzr = DistilBertTokenizer.from_pretrained(model_name)
        def predict_proba(X_test):
            x = X_test.to_list()

            encodings = BertPredict.construct_encodings(x, tkzr, max_len=max_len)
            tfdataset = BertPredict.construct_tfdataset(encodings)
            tfdataset = tfdataset.batch(BertPredict.BATCH_SIZE)

            preds = model.predict(tfdataset).logits
            preds_2 = activations.softmax(tf.convert_to_tensor(preds)).numpy()
            return list(map(lambda x: (np.argmax(x), np.max(preds_2)), preds))

        return predict_proba

"""

if __name__=='__main__':
    with open(preprocessing, 'rb') as f:
        self.preprocessing = pickle.load(f)"""