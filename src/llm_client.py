"""
Ollama LLM client — sends prompts to the Ollama REST API and returns generated text.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))  # seconds — Mistral on CPU can be slow


class LLMClient:
    def __init__(self, base_url: str, model: str = DEFAULT_MODEL) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._http = httpx.Client(timeout=TIMEOUT)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "256")),
                "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "2048")),
            },
        }

        response = self._http.post(f"{self._base_url}/api/generate", json=payload)
        response.raise_for_status()

        data = response.json()
        answer = data.get("response", "").strip()

        logger.info(
            "LLM response received (model=%s, eval_duration=%s)",
            data.get("model"),
            data.get("eval_duration"),
        )
        return answer

    def is_healthy(self) -> bool:
        try:
            resp = self._http.get(f"{self._base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False
