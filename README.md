# Interpretable Narrative Classification System

This project implements a genre classification API for narratives, focusing on interpretability using LIME.

## Features
- **Genre Classification**: Classifies text into genres (Science Fiction, Romance, Mystery, Fantasy) using an SVM model with TF-IDF features.
- **Explainability**: Provides LIME explanations for predictions, highlighting the words that contributed most to the classification.
- **API**: Exposes the model via a FastAPI endpoint with Pydantic validation.
- **Dockerized**: Ready for deployment using Docker.

## Project Structure
- `data/`: Contains the dataset (synthetic).
- `models/`: Serialized models (SVM, Vectorizer).
- `src/`: Source code for preprocessing, features, model, and API.
- `analysis.md`: Analysis of system limitations and future directions.
- `Dockerfile`: Docker configuration.

## Setup

### Local
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python create_dataset.py
   python train.py
   ```
3. Run the API:
   ```bash
   uvicorn src.api.main:app --reload
   ```

### Docker
1. Build the image:
   ```bash
   docker build -t genre-api .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 genre-api
   ```

## API Usage
**Endpoint**: `POST /predict`

**Request**:
```json
{
  "text": "The spaceship landed on the planet.",
  "include_explanation": true
}
```

**Response**:
```json
{
  "genre": "Science Fiction",
  "confidence": 0.85,
  "explanation": [
    ["spaceship", 0.05],
    ["planet", 0.03]
  ]
}
```
