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
import structlog
import logging
from sqlalchemy.orm import Session
from src.database import init_db, SessionLocal, PredictionLog

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

# Global variables for models
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    try:
        logger.info("initializing_database")
        init_db()
        
        logger.info("loading_models")
        classifier = GenreClassifier()
        classifier.load("models/model.joblib")
        
        vectorizer = FeatureExtractor()
        vectorizer.load("models/vectorizer.joblib")
        
        explainer = Explainer(class_names=classifier.model.classes_)
        
        models["classifier"] = classifier
        models["vectorizer"] = vectorizer
        models["explainer"] = explainer
        logger.info("models_loaded_successfully")
    except Exception as e:
        logger.error("error_loading_models", error=str(e))
        raise e
    
    yield
    
    # Clean up on shutdown
    models.clear()


# Initialize Limiter
limiter = Limiter(key_func=get_remote_address)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(title="Genre Classification API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize cache: max 1000 items, 1 hour TTL
prediction_cache = TTLCache(maxsize=1000, ttl=3600)

from fastapi import Depends

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("5/minute")
async def predict(request: Request, prediction_request: PredictionRequest, db: Session = Depends(get_db)):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Create cache key
    cache_key = (prediction_request.text, prediction_request.include_explanation)
    
    # Check cache
    if cache_key in prediction_cache:
        logger.info("cache_hit", text_snippet=prediction_request.text[:20])
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
        logger.info("prediction_complete", genre=response.genre, confidence=response.confidence)
        
        # Log to database
        try:
            db_log = PredictionLog(
                text=prediction_request.text,
                genre=response.genre,
                confidence=response.confidence,
                ip_address=request.client.host,
                model_version="v1"
            )
            db.add(db_log)
            db.commit()
        except Exception as e:
            logger.error("database_log_failed", error=str(e))
            
        return response
        
    except Exception as e:
        logger.error("prediction_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Instrument the app
Instrumentator().instrument(app).expose(app)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount static files
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("src/api/static/index.html")
