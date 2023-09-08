from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class KerasTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_words=None, **kwargs):
        self.num_words = num_words
        self.tokenizer = Tokenizer(num_words=num_words, **kwargs)

    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts(X.values.flatten())
        return self

    def transform(self, X, y=None):
        x = np.array(X.values)
        for i in range(len(x)):
            for j in range(len(x[i])):
                x[i][j] = sum([sum(line) if line else 0 for line in self.tokenizer.texts_to_sequences(x[i][j])])
        return x

    def get_params(self, deep=True):
        return {"num_words": self.num_words}