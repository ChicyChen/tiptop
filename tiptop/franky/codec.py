"""msgpack codec for the ``franky_service`` policy wire.

The driver is an ``openpi_client`` and encodes numpy arrays in openpi's format::

    {b"__ndarray__": <raw bytes>, b"dtype": ..., b"shape": [...]}

``msgpack_numpy`` uses a different, shorter tagging (``b"nd"``, ``b"type"``,
``b"data"``). We decode BOTH so this works against openpi clients and against
``msgpack_numpy``-based test harnesses.

The encoder deliberately binds ``msgpack._cmsgpack`` (the C implementation)
rather than ``msgpack.Packer``: importing ``msgpack_numpy`` anywhere in the
process calls ``msgpack_numpy.patch()``, which rebinds ``msgpack.Packer``
globally and would silently corrupt our replies. This bit us in the cap-x
adapter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # C implementation, immune to msgpack_numpy.patch()
    from msgpack import _cmsgpack as _mp  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - pure-python fallback
    import msgpack as _mp  # type: ignore[no-redef]


def _encode_hook(obj: Any) -> Any:
    """Match ``openpi_client.msgpack_numpy.pack_array`` byte-for-byte.

    The driver decodes with openpi's ``unpack_array``, which keys on
    ``b"__ndarray__"`` being a FLAG and reads the payload from a separate
    ``b"data"`` field. Putting the bytes directly under ``b"__ndarray__"``
    produces a message the driver silently decodes as a 0-d object -- the arm
    then receives no usable action chunk.
    """
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("O", "c"):
            raise ValueError(f"unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    raise TypeError(f"cannot serialise {type(obj)!r}")


def _decode_hook(obj: dict) -> Any:
    if b"__ndarray__" in obj:  # openpi
        return np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"])).reshape(
            obj[b"shape"]
        )
    if b"__npgeneric__" in obj:  # openpi scalar
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    if b"nd" in obj:  # msgpack_numpy
        return np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"type"])).reshape(
            obj[b"shape"]
        )
    return obj


def packb(data: Any) -> bytes:
    # _cmsgpack exposes Packer/Unpacker (and unpackb) but no packb, so build a
    # Packer per call. Binding the class here rather than msgpack.Packer is the
    # whole point: msgpack_numpy.patch() rebinds the latter process-wide.
    return _mp.Packer(default=_encode_hook, use_bin_type=True).pack(data)


def unpackb(raw: bytes) -> Any:
    return _mp.unpackb(
        raw, object_hook=_decode_hook, raw=False, strict_map_key=False
    )
