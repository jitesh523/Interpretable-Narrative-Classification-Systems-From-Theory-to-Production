# Interpretable Narrative Classification System

This project implements a genre classification API for narratives, focusing on interpretability using LIME.

## Features
- **Genre Classification**: Classifies text into genres (Science Fiction, Romance, Mystery, Fantasy) using an SVM model with TF-IDF features.
- **Explainability**: Provides LIME explanations for predictions, highlighting the words that contributed most to the classification.
- **Enterprise Features**:
    - **Database Integration**: Predictions are logged to a SQLite database (`app.db`).
    - **Active Learning Feedback Loop**: Users can submit feedback on predictions via the `/feedback` endpoint.
    - **API Security**: JWT-based authentication protects sensitive endpoints.
    - **External LLM Integration**: The `/explain_llm` endpoint provides natural language explanations using OpenAI (requires `OPENAI_API_KEY`).
- **Production Ready**: Dockerized, cached, rate-limited, and monitored (Prometheus).

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

## Usage

### 1. Basic Prediction
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

### 2. Authentication
Get a token (default credentials: admin/secret):
```bash
curl -X POST "http://localhost:8000/token" -d "username=admin&password=secret"
```

### 3. Submit Feedback
```bash
curl -X POST "http://localhost:8000/feedback" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"prediction_id": 1, "actual_genre": "Mystery", "comments": "Correct!"}'
```

### 4. LLM Explanation
```bash
curl -X POST "http://localhost:8000/explain_llm" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"text": "The detective found a clue.", "include_explanation": false}'
```

## Project Structure
- `data/`: Contains the dataset (synthetic).
- `models/`: Serialized models (SVM, Vectorizer).
- `src/`: Source code for preprocessing, features, model, and API.
- `analysis.md`: Analysis of system limitations and future directions.
- `Dockerfile`: Docker configuration.
