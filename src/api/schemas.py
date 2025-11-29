from pydantic import BaseModel, Field
from typing import List, Tuple, Optional

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, example="The spaceship landed on the planet.")
    include_explanation: bool = Field(False, example=True)

class PredictionResponse(BaseModel):
    genre: str
    confidence: float
    explanation: Optional[List[Tuple[str, float]]] = None
    prediction_id: Optional[int] = None # Added to link feedback

class FeedbackRequest(BaseModel):
    prediction_id: int
    actual_genre: str
    comments: Optional[str] = None
