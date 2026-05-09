"""Patch TiPToP's unavailable Gemini Robotics model to an available Gemini model.

TiPToP upstream defaults to `gemini-robotics-er-1.5-preview`, which returned
404 with the provided key. This patch keeps canonical Google GenAI/Gemini usage
but changes only the model id to `gemini-2.5-flash` unless overridden.
"""
from __future__ import annotations
import os


def apply_patch(model_id: str | None = None):
    model_id = model_id or os.environ.get('TIPTOP_GEMINI_MODEL', 'gemini-2.5-flash')
    import tiptop.perception.gemini as g
    orig_sync = g.detect_and_translate
    orig_async = g.detect_and_translate_async

    def detect_and_translate(image, task_instruction, client=None, model_id_override=None, temperature=None):
        return orig_sync(image, task_instruction, client=client, model_id=model_id_override or model_id, temperature=temperature)

    async def detect_and_translate_async(image, task_instruction, client=None, model_id_override=None, temperature=None):
        return await orig_async(image, task_instruction, client=client, model_id=model_id_override or model_id, temperature=temperature)

    g.detect_and_translate = detect_and_translate
    g.detect_and_translate_async = detect_and_translate_async
