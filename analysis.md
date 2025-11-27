# Analysis of Limitations and Future Directions

## Limitations

### 1. Synthetic Data
The current model is trained on a small synthetic dataset. This limits its ability to generalize to real-world narratives. The vocabulary and sentence structures in the synthetic data are simplistic and do not reflect the complexity of actual literary works.

### 2. Model Simplicity
We are using a Support Vector Machine (SVM) with TF-IDF features. While interpretable and effective for simple baselines, it may struggle with:
- **Contextual Nuances**: TF-IDF ignores word order and context beyond n-grams.
- **Semantic Meaning**: It does not capture semantic similarities between words (e.g., "king" and "queen").

### 3. LIME Instability
LIME explanations can be unstable, meaning that slight perturbations in the input or the random seed can lead to different explanations for the same prediction. This can undermine trust in the system.

### 4. Concurrency Handling
The current API implementation uses `async def` for the prediction endpoint, but the underlying model prediction is a blocking CPU-bound operation. This blocks the FastAPI event loop, potentially degrading performance under high load.

## Future Directions

### 1. Advanced Models
- **Transformer-based Models**: Fine-tune pre-trained models like BERT or RoBERTa for better performance on complex narratives.
- **Embeddings**: Use Word2Vec or GloVe embeddings to capture semantic meaning.

### 2. Enhanced Explainability
- **SHAP (SHapley Additive exPlanations)**: Implement SHAP for more consistent and theoretically grounded explanations.
- **Anchor**: Use Anchor explanations for high-precision rules.

### 3. Production Optimizations
- **Asynchronous Processing**: Offload model predictions to a separate thread pool or a task queue (e.g., Celery) to prevent blocking the event loop.
- **Caching**: Implement caching for frequent predictions to reduce latency.
- **Monitoring**: Add Prometheus/Grafana monitoring to track API performance and model drift.

### 4. Data Augmentation
- Acquire real-world datasets (e.g., CMU Book Summary Dataset).
- Use data augmentation techniques (back-translation, synonym replacement) to increase dataset size and diversity.
