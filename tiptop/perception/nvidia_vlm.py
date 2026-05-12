"""NVIDIA inference API backend for TiPToP's detect-and-translate VLM call.

Drop-in replacements for tiptop.perception.gemini.detect_and_translate{,
_async} that route through NVIDIA's OpenAI-compatible chat-completions
endpoint instead of Google GenAI. Lets us reuse the same NVIDIA_API_KEY
we use for cap-x / vlm-orchestrator and skip the Gemini API key entirely.

Defaults:
- model    : aws/anthropic/bedrock-claude-opus-4-7    (multimodal, JSON-friendly)
- endpoint : https://inference-api.nvidia.com/v1      (matches cap-x ensemble)
- auth     : NVIDIA_API_KEY env var (falls back to OPENAI_API_KEY)

Switch backends without editing this file: set TIPTOP_LLM_BACKEND=nvidia
+ apply_nvidia_patch() in the runner.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from functools import cache

from PIL import Image

from tiptop.perception.gemini import _parse_response, load_prompt

_log = logging.getLogger(__name__)

NVIDIA_BASE_URL = os.environ.get(
    "TIPTOP_NVIDIA_BASE_URL", "https://inference-api.nvidia.com/v1"
)
# Default to the GCP-routed Gemini 2.5 Flash — the tiptop maintainer's
# own fallback when his gemini-robotics-er-1.5 key 404'd. The actual
# robotics-ER preview model isn't exposed on NVIDIA inference; this is
# the closest paper-faithful choice reachable with NVIDIA_API_KEY.
NVIDIA_MODEL = os.environ.get(
    "TIPTOP_NVIDIA_MODEL", "gcp/google/gemini-2.5-flash"
)


def _api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "NVIDIA_API_KEY (or OPENAI_API_KEY) not set; cannot use nvidia VLM backend"
        )
    return key


@cache
def _sync_client():
    from openai import OpenAI

    return OpenAI(api_key=_api_key(), base_url=NVIDIA_BASE_URL)


@cache
def _async_client():
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=_api_key(), base_url=NVIDIA_BASE_URL)


def _encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_messages(image: Image.Image, task_instruction: str) -> list[dict]:
    prompt = load_prompt("detect_and_translate").format(
        task_instruction=task_instruction
    )
    img_b64 = _encode_image(image)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ],
        }
    ]


def _build_kwargs(model_id: str, image: Image.Image,
                  task_instruction: str, temperature: float | None) -> dict:
    """Some NVIDIA models (Claude Opus 4.7) reject the `temperature` field
    entirely; only pass it when explicitly requested.

    max_tokens: 4096 truncated Gemini-2.5-flash mid-response on busier
    LH-CS scenes (6+ objects → JSON > 4 KB). 16384 leaves plenty of
    headroom and is well within Gemini-2.5-flash's 65k output cap.
    """
    kwargs: dict = {
        "model": model_id,
        "messages": _build_messages(image, task_instruction),
        "max_tokens": int(os.environ.get("TIPTOP_NVIDIA_MAX_TOKENS", "16384")),
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


def detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    client=None,  # unused — kept for signature compatibility with gemini.py
    model_id: str = NVIDIA_MODEL,
    temperature: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """NVIDIA-routed detect-and-translate. Sync."""
    response = _sync_client().chat.completions.create(
        **_build_kwargs(model_id, image, task_instruction, temperature)
    )
    return _parse_response(response.choices[0].message.content)


async def detect_and_translate_async(
    image: Image.Image,
    task_instruction: str,
    client=None,
    model_id: str = NVIDIA_MODEL,
    temperature: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """NVIDIA-routed detect-and-translate. Async."""
    response = await _async_client().chat.completions.create(
        **_build_kwargs(model_id, image, task_instruction, temperature)
    )
    return _parse_response(response.choices[0].message.content)
