import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_text
from src.features import FeatureExtractor
from src.model import GenreClassifier
import os

def train():
    # Load data
    df = pd.read_csv("data/dataset.csv")
    
    # Preprocess
    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['genre'], test_size=0.2, random_state=42
    )
    
    # Feature Extraction
    print("Extracting features...")
    feature_extractor = FeatureExtractor()
    X_train_vec = feature_extractor.fit_transform(X_train)
    X_test_vec = feature_extractor.transform(X_test)
    
    # Model Training
    print("Training model...")
    classifier = GenreClassifier()
    classifier.fit(X_train_vec, y_train)
    
    # Evaluation
    print("Evaluating model...")
    results = classifier.evaluate(X_test_vec, y_test)
    print(f"Accuracy: {results['accuracy']}")
    print(results['report'])
    
    # Save artifacts
    print("Saving artifacts...")
    os.makedirs("models", exist_ok=True)
    feature_extractor.save("models/vectorizer.joblib")
    classifier.save("models/model.joblib")
    print("Done.")

if __name__ == "__main__":
    train()
