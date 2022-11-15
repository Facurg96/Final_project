
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from gensim.models import Word2Vec, KeyedVectors, FastText, Doc2Vec
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from text_normalizer import normalize_corpus

class Word2VecProcessor(TransformerMixin):
    def fit(self, X, y=None):
        model_w2v = Word2Vec(X, vector_size=100, workers=6, epochs=50, min_count=1)  
        model_w2v.save('w2vec_model.txt')
        
        return self

    def transform(self, X):
        model = Word2Vec.load('w2vec_model.txt')
        corpus_vectors=[]
        for element in X:
            main_vector=np.zeros([100,],)
            for word in element:
                sub_vector=model.wv[word]
                main_vector=main_vector+ sub_vector
            main_vector=main_vector/len(element)
            corpus_vectors.append(main_vector)
        return corpus_vectors

class FastTextProcessor(TransformerMixin):
    def fit(self, X, y=None):
        model_ft = FastText(X, vector_size=100, workers=6, epochs=50, min_count=1)  
        model_ft.save('fasttext_model.txt')
        
        return self

    def transform(self, X):
        model = FastText.load('fasttext_model.txt')
        corpus_vectors=[]
        for element in X:
            main_vector=np.zeros([100,],)
            for word in element:
                sub_vector=model.wv[word]
                main_vector=main_vector+ sub_vector
            main_vector=main_vector/len(element)
            corpus_vectors.append(main_vector)
        return corpus_vectors


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
    def transform(self, X):
        #X["name"] = X["name"].fillna(X["description"])
        X["name"] = X["name"].fillna("UNK")
        X["description"] = X["description"].fillna("UNK")
        return X

    def fit(self, X_param, y=None):
        return self

class NameDescriptionNormalization(TransformerMixin):
    def transform(self, X):
        X["name"] = normalize_corpus(list(X["name"]))
        X["description"] = normalize_corpus(list(X["description"]))
        return X

    def fit(self, X, y=None):
        return self

class TfidfVectorizerTransformer(TransformerMixin):
    def transform(self, X):
        return self.tfidf.transform(list(X.iloc[:,0]))

    def fit(self, X, y=None):
        self.tfidf = TfidfVectorizer(norm=None).fit(list(X.iloc[:,0]))
        return self

class OrdinalEncoding(TransformerMixin):
    def fit(self, X, y=None):
        self.enc = OrdinalEncoder(dtype='int')
        self.enc.fit(X)
        return self
    
    def transform(self, X):
        aux = []
        X = np.array(X)
        X = X.reshape(-1,1)
        aux = self.enc.transform(X)
        #aux.value_counts()
        return aux

