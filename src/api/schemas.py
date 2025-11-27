from pydantic import BaseModel, Field
from typing import List, Tuple

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, example="The spaceship landed on the planet.")
    include_explanation: bool = Field(False, example=True)

class PredictionResponse(BaseModel):
    genre: str
    confidence: float
    explanation: List[Tuple[str, float]] = []
