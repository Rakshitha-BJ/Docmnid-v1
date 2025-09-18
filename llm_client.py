import os
import logging
from typing import Dict, List, Optional
import requests
import re

from prompt_templates import build_prompt

logger = logging.getLogger("docmind.llm")


class GeminiClient:
    """Adapter for Gemini 2.x Flash via HTTP API.

    Uses `requests` to call the Google Generative Language API. Update MODEL_NAME
    if you wish to switch to another Gemini model.
    """

    # Central place to switch models (e.g., "gemini-2.5-flash")
    MODEL_NAME = "gemini-2.5-flash"

    def __init__(self, api_key_env: str = "GEMINI_API_KEY", timeout_s: int = 20):
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env)
        self.timeout_s = timeout_s
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set. LLM calls will fail until provided.")
        # Endpoint for the selected model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL_NAME}:generateContent"

    def generate_answer(self, question: str, contexts: List[Dict]) -> str:
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

        text = self._normalize_citations(text, contexts)
        return text

    def _generate(self, prompt: str) -> Dict:
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        body = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
            },
        }

        def do_request():
            resp = requests.post(self.base_url, headers=headers, params=params, json=body, timeout=self.timeout_s)
            if resp.status_code == 429:
                raise RuntimeError(f"Rate limited: {resp.text}")
            resp.raise_for_status()
            return resp.json()

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

    @staticmethod
    def _normalize_citations(text: str, contexts: List[Dict]) -> str:
        def repl_num(m):
            try:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(contexts):
                    return f"[Doc: page {contexts[idx]['page']}]"
            except Exception:
                pass
            return ""

        text = re.sub(r"\[(\d+)\]", repl_num, text)

        allowed_pages = {c.get("page") for c in contexts}

        def validate_page(m):
            try:
                p = int(m.group(1))
                return m.group(0) if p in allowed_pages else ""
            except Exception:
                return ""

        text = re.sub(r"\[Doc:\s*page\s*(\d+)\]", validate_page, text, flags=re.IGNORECASE)
        return text
