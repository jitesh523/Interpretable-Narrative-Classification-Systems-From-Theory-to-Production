import os
from openai import OpenAI
import structlog

logger = structlog.get_logger()

class LLMExplainer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("openai_api_key_missing", message="LLM features will be disabled or mocked.")

    def generate_explanation(self, text: str, predicted_genre: str, confidence: float) -> str:
        if not self.client:
            return "LLM explanation unavailable (API key missing). This is a mock explanation: The text contains keywords typical of this genre."
        
        try:
            prompt = f"""
            Analyze the following text and explain why it belongs to the genre '{predicted_genre}' (Confidence: {confidence:.2f}).
            Text: "{text}"
            Keep the explanation concise (under 50 words).
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a literary expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e))
            return "Failed to generate LLM explanation."
