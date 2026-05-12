"""Patch TiPToP's Gemini model id (no-op routing through Gemini).

TiPToP upstream code originally called `gemini-robotics-er-1.5-preview`.
Google retired 1.5 on 2026-05-12 (404 with "no longer available. Please
update your code to use a newer model"); 1.6 is the current preview
successor and is verified accessible from our Google AI Studio key.
Override via the `TIPTOP_GEMINI_MODEL` env var if needed (e.g.
`gemini-2.5-flash` for a fallback).
"""
from __future__ import annotations
import os


def apply_patch(model_id: str | None = None):
    model_id = model_id or os.environ.get(
        'TIPTOP_GEMINI_MODEL', 'gemini-robotics-er-1.6-preview',
    )
    import tiptop.perception.gemini as g
    orig_sync = g.detect_and_translate
    orig_async = g.detect_and_translate_async

    def detect_and_translate(image, task_instruction, client=None, model_id_override=None, temperature=None):
        return orig_sync(image, task_instruction, client=client, model_id=model_id_override or model_id, temperature=temperature)

    async def detect_and_translate_async(image, task_instruction, client=None, model_id_override=None, temperature=None):
        return await orig_async(image, task_instruction, client=client, model_id=model_id_override or model_id, temperature=temperature)

    g.detect_and_translate = detect_and_translate
    g.detect_and_translate_async = detect_and_translate_async
