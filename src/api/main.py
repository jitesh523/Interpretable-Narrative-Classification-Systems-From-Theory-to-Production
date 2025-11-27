from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.api.schemas import PredictionRequest, PredictionResponse
from src.model import GenreClassifier
from src.features import FeatureExtractor
from src.explainability import Explainer
from src.preprocessing import clean_text
import numpy as np
import os

# Global variables for models
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    try:
        print("Loading models...")
        classifier = GenreClassifier()
        classifier.load("models/model.joblib")
        
        vectorizer = FeatureExtractor()
        vectorizer.load("models/vectorizer.joblib")
        
        explainer = Explainer(class_names=classifier.model.classes_)
        
        models["classifier"] = classifier
        models["vectorizer"] = vectorizer
        models["explainer"] = explainer
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e
    
    yield
    
    # Clean up on shutdown
    models.clear()

app = FastAPI(title="Genre Classification API", lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Preprocess
        cleaned_text = clean_text(request.text)
        
        # Vectorize
        vectorizer = models["vectorizer"]
        vec = vectorizer.transform([cleaned_text])
        
        # Predict
        classifier = models["classifier"]
        prediction_idx = classifier.predict(vec)[0]
        probabilities = classifier.predict_proba(vec)[0]
        
        # Get confidence
        # prediction_idx is the class label (string), we need to find its index in classes_ to get probability
        class_index = list(classifier.model.classes_).index(prediction_idx)
        confidence = float(probabilities[class_index])
        
        explanation = []
        if request.include_explanation:
            explainer = models["explainer"]
            # LIME requires a function that takes raw text and returns probabilities
            def predict_proba_fn(texts):
                clean_texts = [clean_text(t) for t in texts]
                vecs = vectorizer.transform(clean_texts)
                return classifier.predict_proba(vecs)
            
            explanation = explainer.explain_instance(request.text, predict_proba_fn)
            # Convert numpy types to native python types for JSON serialization
            explanation = [(str(k), float(v)) for k, v in explanation]
            
        return PredictionResponse(
            genre=prediction_idx,
            confidence=confidence,
            explanation=explanation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
