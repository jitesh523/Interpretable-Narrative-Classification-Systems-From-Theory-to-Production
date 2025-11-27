FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies
# Install build tools for scikit-image/lime
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt stopwords wordnet punkt_tab && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
