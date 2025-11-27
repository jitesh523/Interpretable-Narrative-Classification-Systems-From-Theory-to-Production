from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

class FeatureExtractor:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        
    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        return self.vectorizer.transform(texts)
    
    def save(self, path):
        joblib.dump(self.vectorizer, path)
        
    def load(self, path):
        if os.path.exists(path):
            self.vectorizer = joblib.load(path)
        else:
            raise FileNotFoundError(f"Vectorizer not found at {path}")
