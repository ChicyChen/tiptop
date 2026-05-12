"""Patch TiPToP's Gemini model id (no-op routing through Gemini).

TiPToP upstream code calls `gemini-robotics-er-1.5-preview` directly.
This patch keeps that paper-faithful default but lets us override via the
`TIPTOP_GEMINI_MODEL` env var (e.g. to fall back to `gemini-2.5-flash`
if the robotics-ER preview returns 404 for the provided key).
"""
from __future__ import annotations
import os


def apply_patch(model_id: str | None = None):
    model_id = model_id or os.environ.get(
        'TIPTOP_GEMINI_MODEL', 'gemini-robotics-er-1.5-preview',
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
