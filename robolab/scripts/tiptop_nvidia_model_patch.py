"""Patch TiPToP's Gemini detect-and-translate calls to use the NVIDIA
inference API (OpenAI-compatible) instead.

Same idea as `tiptop_gemini_model_patch.py`: monkey-patch
``tiptop.perception.gemini.detect_and_translate{,_async}`` so that runners
which call `from tiptop.perception.gemini import detect_and_translate*`
transparently route through NVIDIA. No edits to upstream tiptop files
required.

Activated when TIPTOP_LLM_BACKEND=nvidia is set in the environment, or
by importing and calling ``apply_patch()`` directly.

Auth: reads NVIDIA_API_KEY (or OPENAI_API_KEY) from the environment.
Model override: TIPTOP_NVIDIA_MODEL (default aws/anthropic/bedrock-claude-opus-4-7).
Endpoint override: TIPTOP_NVIDIA_BASE_URL (default https://inference-api.nvidia.com/v1).
"""
from __future__ import annotations


def apply_patch() -> None:
    import tiptop.perception.gemini as g
    from tiptop.perception import nvidia_vlm

    g.detect_and_translate = nvidia_vlm.detect_and_translate
    g.detect_and_translate_async = nvidia_vlm.detect_and_translate_async
