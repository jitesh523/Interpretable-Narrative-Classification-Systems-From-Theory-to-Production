from src.model import GenreClassifier
from src.features import FeatureExtractor
from src.explainability import Explainer
from src.preprocessing import clean_text
import numpy as np

def verify_lime():
    print("Loading model and vectorizer...")
    classifier = GenreClassifier()
    classifier.load("models/model.joblib")
    
    feature_extractor = FeatureExtractor()
    feature_extractor.load("models/vectorizer.joblib")
    
    print("Initializing explainer...")
    explainer = Explainer(class_names=classifier.model.classes_)
    
    text = "The spaceship landed on the planet."
    print(f"Explaining text: '{text}'")
    
    def predict_proba(texts):
        clean_texts = [clean_text(t) for t in texts]
        vecs = feature_extractor.transform(clean_texts)
        return classifier.predict_proba(vecs)
    
    exp = explainer.explain_instance(text, predict_proba)
    print("Explanation generated successfully.")
    print("Top features:", exp)

if __name__ == "__main__":
    verify_lime()
