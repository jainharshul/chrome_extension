import re
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for text in X:
            if pd.isna(text):
                text = ""
            num_chars = len(text)
            num_words = len(text.split())
            num_exclaims = text.count('!')
            num_questions = text.count('?')
            num_http = len(re.findall(r'http[s]?://', text))
            num_dollars = text.count('$')
            num_digits = sum(c.isdigit() for c in text)
            feats.append([num_chars, num_words, num_exclaims, num_questions,
                          num_http, num_dollars, num_digits])
        return np.array(feats)


class CombinedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=20000):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=3
        )
        self.stats = TextStats()

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        self.stats.fit(X)
        return self

    def transform(self, X):
        tfidf_features = self.tfidf.transform(X)
        stats_features = csr_matrix(self.stats.transform(X))
        # horizontally stack sparse and dense features
        return hstack([tfidf_features, stats_features])