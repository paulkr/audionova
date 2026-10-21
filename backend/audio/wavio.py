"""WAV / RF64 reading, mirroring ``engine/include/anv/io/WavFile.h``.

WHY NOT ``soundfile``. It wraps libsndfile, which is LGPL — the exact
dependency the C++ engine was written to avoid, and re-introducing it on the
Python side would put the obligation straight back into the product. It is also
a parser exposed to bytes we did not create.

WHY NOT ``scipy.io.wavfile``. It is BSD and would be fine on licence grounds,
but it and the engine would then be two independent opinions about what a WAV
file means. The failure mode of that disagreement is not a crash: it is a
measurement report computed from subtly different samples than the ones the
renderer processed, which makes every number in the report quietly wrong about
the audio the user will actually hear. One decoder semantic, expressed twice
and tested against each other, is worth eighty lines.

Sample values come back as float64 in [-1, 1] for integer formats, and
unclamped for float formats — floats can legitimately exceed full scale and
clamping on read would destroy headroom that was left deliberately.

Layout is CHANNELS-FIRST (shape ``(channels, frames)``), matching the engine's
planar buffers. Interleaved is a real-time convention with no offline benefit.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

_FORMAT_PCM = 0x0001
_FORMAT_FLOAT = 0x0003
_FORMAT_EXTENSIBLE = 0xFFFE


class WavError(Exception):
    """Raised for anything this reader will not accept."""


@dataclass
class AudioFile:
    samples: np.ndarray  # float64, shape (channels, frames)
    sample_rate: float
    bit_depth: int
    is_float: bool

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def format_name(self) -> str:
        return f"{self.bit_depth}-bit {'float' if self.is_float else 'PCM'}"


def decode_wav(data: bytes, *, allow_non_finite: bool = False) -> AudioFile:
    """Decode WAV or RF64 bytes.

    Every length in the file is clamped against the bytes actually present
    before it is used. The ingest sandbox means this should never see hostile
    input, but "should never" is not a bounds check.
    """
    if len(data) < 12:
        raise WavError("file is too short to contain a RIFF header")

    riff, wave = data[0:4], data[8:12]
    is_rf64 = riff == b"RF64"
    if not is_rf64 and riff != b"RIFF":
        raise WavError(f"not a WAV file (expected RIFF or RF64, got {riff!r})")
    if wave != b"WAVE":
        raise WavError(f"not a WAV file (expected WAVE, got {wave!r})")

    pos = 12
    ds64_data_size = 0
    have_ds64 = False
    fmt = None
    payload = None

    while pos + 8 <= len(data):
        tag = data[pos : pos + 4]
        declared = struct.unpack_from("<I", data, pos + 4)[0]
        pos += 8

        size = declared
        # RF64 stores the real 64-bit size in ds64 and leaves a sentinel here.
        if tag == b"data" and is_rf64 and declared == 0xFFFFFFFF:
            size = ds64_data_size
        size = min(size, len(data) - pos)

        if tag == b"ds64":
            if size < 16:
                raise WavError("ds64 chunk is too short")
            ds64_data_size = struct.unpack_from("<Q", data, pos + 8)[0]
            have_ds64 = True
        elif tag == b"fmt ":
            if size < 16:
                raise WavError("fmt chunk is too short")
            format_tag, channels, sample_rate, _, _, bits = struct.unpack_from(
                "<HHIIHH", data, pos
            )
            if format_tag == _FORMAT_EXTENSIBLE:
                if size < 40:
                    raise WavError("WAVE_FORMAT_EXTENSIBLE fmt chunk is too short")
                # The subformat GUID's first two bytes carry the real tag.
                format_tag = struct.unpack_from("<H", data, pos + 24)[0]
            fmt = (format_tag, channels, sample_rate, bits)
        elif tag == b"data":
            payload = data[pos : pos + size]

        pos += size
        # Chunks are word-aligned; an odd size is followed by a pad byte that is
        # NOT counted in the size field. Forgetting it misaligns every chunk
        # after this one, which is the classic WAV parser bug.
        if size & 1:
            pos += 1

    if fmt is None:
        raise WavError("no fmt chunk")
    if payload is None:
        raise WavError("no data chunk")
    if is_rf64 and not have_ds64:
        raise WavError("RF64 file with no ds64 chunk")

    format_tag, channels, sample_rate, bits = fmt
    if channels < 1:
        raise WavError("file declares zero channels")
    if sample_rate < 1:
        raise WavError("file declares a zero sample rate")

    if format_tag == _FORMAT_PCM:
        samples = _decode_pcm(payload, channels, bits)
        is_float = False
    elif format_tag == _FORMAT_FLOAT:
        samples = _decode_float(payload, channels, bits)
        # The engine's reader refuses these too, and the two must agree on
        # what a file contains. A NaN reaches the measurement report as NaN
        # and the render as a dropout.
        bad = np.argwhere(~np.isfinite(samples)) if not allow_non_finite else []
        if len(bad):
            channel, frame = (int(v) for v in bad[0])
            raise WavError(f"sample {frame} of channel {channel} is not finite "
                           "(NaN or infinity)")
        is_float = True
    else:
        raise WavError(
            f"unsupported WAV format tag 0x{format_tag:04x} — only uncompressed "
            "PCM and float are read here; everything else goes through the "
            "ingest decoder"
        )

    return AudioFile(
        samples=samples,
        sample_rate=float(sample_rate),
        bit_depth=int(bits),
        is_float=is_float,
    )


def _whole(payload: bytes, size: int) -> bytes:
    """Drop a trailing partial sample, which `np.frombuffer` refuses outright.

    Only the 24-bit path trimmed, so a file truncated mid-sample decoded at 24
    bits and raised ValueError — not WavError — at every other depth, which the
    API reported as a 500 rather than reading what was there.
    """
    return payload[: len(payload) - len(payload) % size]


def _decode_pcm(payload: bytes, channels: int, bits: int) -> np.ndarray:
    # Scaling is by 2^(n-1), matching the engine exactly. Two's complement runs
    # from -2^(n-1) to +2^(n-1)-1, so this maps the most negative code to
    # exactly -1.0 and the most positive to 1.0 - step. Dividing by 2^(n-1)-1
    # instead would let full-scale negative read as -1.00003, which then clips
    # on the way back out.
    if bits == 16:
        flat = np.frombuffer(_whole(payload, 2), dtype="<i2").astype(np.float64) / 32768.0
    elif bits == 24:
        raw = np.frombuffer(payload, dtype=np.uint8)
        usable = (len(raw) // 3) * 3
        triples = raw[:usable].reshape(-1, 3).astype(np.int32)
        packed = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        # Sign-extend the 24-bit value into 32 bits.
        packed = np.where(packed >= 0x800000, packed - 0x1000000, packed)
        flat = packed.astype(np.float64) / 8388608.0
    elif bits == 32:
        flat = np.frombuffer(_whole(payload, 4), dtype="<i4").astype(np.float64) / 2147483648.0
    elif bits == 8:
        # 8-bit WAV is UNSIGNED, alone among the integer depths. Reading it as
        # signed produces audio that is inverted and offset by half scale, and
        # it still looks like a waveform.
        flat = (np.frombuffer(payload, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise WavError(f"unsupported PCM depth: {bits} bits")
    return _deinterleave(flat, channels)


def _decode_float(payload: bytes, channels: int, bits: int) -> np.ndarray:
    if bits == 32:
        flat = np.frombuffer(_whole(payload, 4), dtype="<f4").astype(np.float64)
    elif bits == 64:
        flat = np.frombuffer(_whole(payload, 8), dtype="<f8").astype(np.float64)
    else:
        raise WavError(f"unsupported float depth: {bits} bits")
    return _deinterleave(flat, channels)


def _deinterleave(flat: np.ndarray, channels: int) -> np.ndarray:
    frames = len(flat) // channels
    if frames == 0:
        return np.zeros((channels, 0), dtype=np.float64)
    # Drop a trailing partial frame rather than raising: a file truncated
    # mid-frame is still worth measuring, and the alternative is refusing to
    # analyse audio that plays fine everywhere else.
    return np.ascontiguousarray(flat[: frames * channels].reshape(frames, channels).T)


def read_wav(path: str, *, allow_non_finite: bool = False) -> AudioFile:
    with open(path, "rb") as handle:
        return decode_wav(handle.read(), allow_non_finite=allow_non_finite)


def encode_wav(
    samples: np.ndarray, sample_rate: float, bit_depth: int = 24, is_float: bool = False
) -> bytes:
    """Encode ``(channels, frames)`` float64 to WAV bytes.

    Present so tests can build fixtures and round-trip against the engine. The
    renderer writes the files that ship; this is for the analysis side.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    channels, frames = samples.shape

    if is_float:
        dtype = "<f4" if bit_depth == 32 else "<f8"
        # Float output is deliberately NOT clamped.
        interleaved = samples.T.astype(dtype)
        format_tag, bytes_per = _FORMAT_FLOAT, 4 if bit_depth == 32 else 8
    else:
        scale = float(1 << (bit_depth - 1))
        clipped = np.clip(np.nan_to_num(samples, nan=0.0), -1.0, 1.0)
        q = np.clip(np.round(clipped * scale), -scale, scale - 1).astype(np.int64)
        if bit_depth == 16:
            interleaved = q.T.astype("<i2")
            bytes_per = 2
        elif bit_depth == 24:
            as32 = (q.T.astype(np.int64) & 0xFFFFFF).astype("<u4")
            interleaved = np.ascontiguousarray(as32).view(np.uint8).reshape(-1, 4)[:, :3]
            bytes_per = 3
        elif bit_depth == 32:
            interleaved = q.T.astype("<i4")
            bytes_per = 4
        else:
            raise WavError(f"cannot write {bit_depth}-bit PCM")
        format_tag = _FORMAT_PCM

    payload = interleaved.tobytes()

    # Extensible above two channels or two bytes per sample, matching the
    # engine: plain WAVE_FORMAT_PCM with 24-bit samples is out of spec and some
    # hosts read it as noise.
    extensible = channels > 2 or bytes_per > 2
    fmt_size = 40 if extensible else 16

    chunks = b"fmt " + struct.pack(
        "<IHHIIHH",
        fmt_size,
        _FORMAT_EXTENSIBLE if extensible else format_tag,
        channels,
        int(round(sample_rate)),
        int(round(sample_rate)) * channels * bytes_per,
        channels * bytes_per,
        bytes_per * 8,
    )
    if extensible:
        chunks += struct.pack("<HHIH", 22, bytes_per * 8, 0, format_tag)
        chunks += bytes(
            (0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80,
             0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71)
        )
    chunks += b"data" + struct.pack("<I", len(payload)) + payload
    if len(payload) & 1:
        chunks += b"\x00"

    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def write_wav(
    path: str,
    samples: np.ndarray,
    sample_rate: float,
    bit_depth: int = 24,
    is_float: bool = False,
) -> None:
    with open(path, "wb") as handle:
        handle.write(encode_wav(samples, sample_rate, bit_depth, is_float))
