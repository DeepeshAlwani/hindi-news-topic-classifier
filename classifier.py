"""
Core classification logic: sends an image to a vision-capable model on
OpenRouter and returns a structured topic classification.

Setup:
    pip install openai
    export OPENROUTER_API_KEY="sk-or-..."
"""

import base64
import json
import mimetypes
import os
import time

from openai import OpenAI

# ---- CONFIG -----------------------------------------------------------
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

TAXONOMY = [
    "Politics",
    "Sports",
    "Business/Economy",
    "Entertainment",
    "Crime",
    "International",
    "Education/Exams",   # govt exams, recruitment, results, paper leaks — common IN news category
    "Technology",
    "Health",
    "Other",
]

SYSTEM_PROMPT = f"""You are analyzing a screenshot of an Indian news broadcast or news website. \
The text in the image may be in Hindi (Devanagari script), English, or a mix of both. \
The image may contain unrelated advertisements, product packaging, or brand logos (for example, \
spice packets, banner ads, sponsor logos) — completely ignore these and do not let them influence \
your classification.

Your job:
1. Read and transcribe the actual news content visible in the image — headlines, ticker text, \
quoted tweets/posts, breaking news banners, or on-screen captions.
2. Classify the core news story into exactly one of these categories:
   {", ".join(TAXONOMY)}
3. If multiple unrelated news items appear, pick the most prominent one \
(largest text / breaking news banner / main headline).
4. Use "Other" only if the story genuinely does not fit any other category — do not default to \
it just because the story is short or partially cropped.

Respond with ONLY a valid JSON object, no markdown fences, no extra text, in this exact shape:
{{
  "extracted_text": "<the news text you read, transcribed as-is>",
  "extracted_text_translation": "<brief English translation if original was Hindi, else same as extracted_text>",
  "category": "<one label from the taxonomy>",
  "confidence": "high|medium|low",
  "reasoning": "<one sentence explaining the classification>"
}}
"""

USER_PROMPT = "Analyze this image and return the JSON classification as instructed."


def encode_image(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENROUTER_API_KEY in your environment first.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def classify_image(image_path: str, max_retries: int = 3, retry_delay: float = 5.0) -> dict:
    """
    Classify a single image. Retries on transient errors (rate limits, timeouts,
    malformed JSON) with a simple backoff, since free-tier models on OpenRouter
    can be flaky under load.
    """
    client = _get_client()
    image_data_url = encode_image(image_path)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    },
                ],
                temperature=0,
                max_tokens=600,
                extra_headers={
                    "HTTP-Referer": "https://example.com",
                    "X-Title": "hindi-news-classifier",
                },
            )

            raw = response.choices[0].message.content.strip()

            # Some models wrap JSON in ```json fences despite instructions not to.
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

            parsed = json.loads(raw)
            parsed["_model"] = MODEL
            parsed["_image"] = os.path.basename(image_path)
            parsed["_parse_error"] = False
            return parsed

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}. Raw output: {raw[:300]}"
        except Exception as e:  # rate limits, timeouts, connection errors, etc.
            last_error = str(e)

        if attempt < max_retries:
            time.sleep(retry_delay * attempt)  # simple linear backoff

    return {
        "_model": MODEL,
        "_image": os.path.basename(image_path),
        "_parse_error": True,
        "error": last_error,
    }