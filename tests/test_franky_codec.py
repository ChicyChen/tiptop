"""msgpack codec compatibility with the driver's openpi_client.

Two traps, both hit for real in the cap-x adapter:

* ``msgpack_numpy.patch()`` rebinds ``msgpack.Packer`` process-wide, so binding
  that name corrupts our replies as soon as anything imports msgpack_numpy.
* openpi tags arrays ``b"__ndarray__"`` while msgpack_numpy uses ``b"nd"``; the
  decoder must accept both.
"""

from __future__ import annotations

import numpy as np

from tiptop.franky.codec import packb, unpackb


class TestRoundTrip:
    def test_arrays_survive_dtype_and_shape(self):
        for arr in (
            np.zeros((720, 1280, 3), np.uint8),
            np.full((720, 1280), 0.9, np.float32),
            np.arange(7, dtype=np.float64),
        ):
            out = unpackb(packb({"a": arr}))["a"]
            assert out.dtype == arr.dtype
            assert out.shape == arr.shape
            assert np.array_equal(out, arr)

    def test_nan_survives(self):
        d = np.full((4, 4), np.nan, np.float32)
        assert np.isnan(unpackb(packb({"d": d}))["d"]).all()

    def test_mixed_payload(self):
        wire = {"prompt": "pick up the block", "episode_id": 3,
                "q": np.zeros(7)}
        out = unpackb(packb(wire))
        assert out["prompt"] == "pick up the block"
        assert out["episode_id"] == 3
        assert out["q"].shape == (7,)


class TestOpenpiWireCompat:
    def test_decodes_openpi_tagging(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        # openpi's pack_array: __ndarray__ is a FLAG, payload lives in `data`.
        raw = packb(
            {
                "x": {
                    b"__ndarray__": True,
                    b"data": arr.tobytes(),
                    b"dtype": arr.dtype.str,
                    b"shape": [2, 3],
                }
            }
        )
        assert np.array_equal(unpackb(raw)["x"], arr)

    def test_decodes_msgpack_numpy_tagging(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        raw = packb(
            {
                "x": {
                    b"nd": True,
                    b"type": arr.dtype.str,
                    b"shape": [2, 3],
                    b"data": arr.tobytes(),
                }
            }
        )
        assert np.array_equal(unpackb(raw)["x"], arr)

    def test_survives_msgpack_numpy_patching_the_process(self):
        """The failure mode that cost real debugging time in cap-x."""
        import msgpack
        import msgpack_numpy

        msgpack_numpy.patch()                 # rebinds msgpack.Packer globally
        arr = np.full((3, 3), 0.5, np.float32)
        out = unpackb(packb({"a": arr}))["a"]
        assert np.array_equal(out, arr), (
            "codec must bind msgpack._cmsgpack, not msgpack.Packer"
        )


class TestInteropWithARealMsgpackNumpyClient:
    def test_a_msgpack_numpy_encoder_can_be_decoded(self):
        import msgpack
        import msgpack_numpy

        arr = np.full((720, 1280), 0.9, np.float32)
        raw = msgpack.packb(
            {"observation/depth_external": arr},
            default=msgpack_numpy.encode,
            use_bin_type=True,
        )
        out = unpackb(raw)
        assert np.array_equal(out["observation/depth_external"], arr)


class TestByteExactWithOpenpi:
    """Our encoder must match ``openpi_client.msgpack_numpy.pack_array``.

    The driver decodes with openpi's ``unpack_array``, which treats
    ``b"__ndarray__"`` as a FLAG and reads bytes from ``b"data"``. A first
    version put the payload directly under ``b"__ndarray__"``; openpi decoded
    that as a 0-d object, so the arm would have received no usable chunk while
    the server looked healthy.
    """

    def test_encoding_has_openpis_exact_key_set(self):
        arr = np.zeros((8, 8))
        raw = packb({"actions": arr})
        # Decode WITHOUT our hook to inspect the raw tagging on the wire.
        import msgpack

        plain = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        tagged = plain["actions"]
        assert set(tagged) == {
            b"__ndarray__",
            b"data",
            b"dtype",
            b"shape",
        }, f"key set differs from openpi's pack_array: {sorted(tagged)}"
        assert tagged[b"__ndarray__"] is True, "__ndarray__ must be a flag"
        assert isinstance(tagged[b"data"], (bytes, bytearray))

    def test_scalars_use_openpis_npgeneric_tagging(self):
        import msgpack

        raw = packb({"s": np.float32(1.5)})
        plain = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        assert set(plain["s"]) == {b"__npgeneric__", b"data", b"dtype"}

    def test_action_chunk_shape_survives(self):
        """(8, 8) is the driver's expected chunk shape: 7 joints + gripper."""
        chunk = np.arange(64, dtype=np.float64).reshape(8, 8)
        out = unpackb(packb({"actions": chunk}))["actions"]
        assert out.shape == (8, 8)
        assert np.array_equal(out, chunk)
