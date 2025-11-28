from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from src.api.schemas import PredictionRequest, PredictionResponse
from src.model import GenreClassifier
from src.features import FeatureExtractor
from src.explainability import Explainer
from src.preprocessing import clean_text
import numpy as np
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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


# Initialize Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Genre Classification API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize cache: max 1000 items, 1 hour TTL
prediction_cache = TTLCache(maxsize=1000, ttl=3600)

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("5/minute")
async def predict(request: Request, prediction_request: PredictionRequest):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Create cache key
    cache_key = (prediction_request.text, prediction_request.include_explanation)
    
    # Check cache
    if cache_key in prediction_cache:
        return prediction_cache[cache_key]
    
    loop = asyncio.get_event_loop()
    
    def blocking_predict():
        # Preprocess
        cleaned_text = clean_text(prediction_request.text)
        
        # Vectorize
        vectorizer = models["vectorizer"]
        vec = vectorizer.transform([cleaned_text])
        
        # Predict
        classifier = models["classifier"]
        prediction_idx = classifier.predict(vec)[0]
        probabilities = classifier.predict_proba(vec)[0]
        
        # Get confidence
        class_index = list(classifier.model.classes_).index(prediction_idx)
        confidence = float(probabilities[class_index])
        
        explanation = []
        if prediction_request.include_explanation:
            explainer = models["explainer"]
            def predict_proba_fn(texts):
                clean_texts = [clean_text(t) for t in texts]
                vecs = vectorizer.transform(clean_texts)
                return classifier.predict_proba(vecs)
            
            explanation = explainer.explain_instance(prediction_request.text, predict_proba_fn)
            explanation = [(str(k), float(v)) for k, v in explanation]
            
        return PredictionResponse(
            genre=prediction_idx,
            confidence=confidence,
            explanation=explanation
        )

    try:
        # Run blocking code in a separate thread
        response = await loop.run_in_executor(None, blocking_predict)
        
        # Store in cache
        prediction_cache[cache_key] = response
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Instrument the app
Instrumentator().instrument(app).expose(app)
