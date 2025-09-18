import os
import logging
from typing import Dict, List, Optional
import requests

from prompt_templates import build_prompt

logger = logging.getLogger("docmind.llm")


class GeminiClient:
    """Adapter for Gemini 2.0 Flash via HTTP API.

    This class uses `requests` to demonstrate a minimal REST call. Replace the
    `generate` method with the official Google Gemini SDK if preferred.

    Expected environment variable: GEMINI_API_KEY

    References:
    - See Google AI Studio / Gemini API docs for REST endpoints and parameters.
      Example docs: https://ai.google.dev/ (update with the correct endpoint for 2.0 Flash)
    """

    def __init__(self, api_key_env: str = "GEMINI_API_KEY", timeout_s: int = 20):
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env)
        self.timeout_s = timeout_s
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set. LLM calls will fail until provided.")
        # Placeholder: update BASE_URL and model path as per official API spec
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def generate_answer(self, question: str, contexts: List[Dict]) -> str:
        """Generate a grounded answer using provided contexts.

        If contexts are empty, returns an "I don't know" message.
        """
        if not contexts:
            return "I don't know based on the provided document."

        prompt = build_prompt(question, contexts)

        try:
            content = self._generate(prompt)
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            return "I don't know based on the provided document."

        text = self._extract_text(content)
        if not text:
            return "I don't know based on the provided document."
        return text

    def _generate(self, prompt: str) -> Dict:
        """Call Gemini REST API with retries and timeout.

        Note: This is a minimal example. The actual Gemini API expects a specific JSON body
        with content blocks. Adjust as per the latest docs.
        """
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        # Minimal body following the generateContent schema for text-only input.
        body = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 512,
            },
        }

        def do_request():
            resp = requests.post(self.base_url, headers=headers, params=params, json=body, timeout=self.timeout_s)
            if resp.status_code == 429:
                raise RuntimeError(f"Rate limited: {resp.text}")
            resp.raise_for_status()
            return resp.json()

        # Simple exponential backoff
        backoff = 1.0
        for attempt in range(5):
            try:
                return do_request()
            except Exception as e:
                if attempt == 4:
                    raise
                logger.warning("Gemini request failed (attempt %d): %s", attempt + 1, e)
                import time
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Unreachable")

    @staticmethod
    def _extract_text(response_json: Dict) -> Optional[str]:
        """Extract text from Gemini response JSON.

        This method follows the common Gemini REST response structure:
        {
          "candidates": [
            { "content": { "parts": [{"text": "..."}] } }
          ]
        }
        """
        try:
            candidates = response_json.get("candidates") or []
            if not candidates:
                return None
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            if not parts:
                return None
            text = parts[0].get("text")
            return text
        except Exception:
            return None
