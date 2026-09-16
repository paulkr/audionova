"""The chain's explanation, read aloud. ElevenLabs text-to-speech over httpx.

WHAT THIS IS FOR. Every graph carries a plain-language `explanation` of what
the chain does and why (see `backend/schemas/graph.py`). An engineer at a desk
reads it. An engineer with their hands on a fader, or with their eyes on the
meters while the excerpt plays, does not — so the paragraph that justifies the
chain is the one part of the result that ends up not being consumed. This turns
it into a voice track that plays while they listen to the processed excerpt.

WHAT IT IS NOT. Not a conversational agent, not speech-to-text, not a way of
driving the planner. The text goes out, audio comes back, and nothing about the
audio feeds back into any decision. It is a presentation layer for a string
that already exists, which is why it has no planner and no schema of its own.

THE MODEL NEVER SEES THE AUDIO, AND THE VOICE NEVER SEES THE MEASUREMENTS. The
only thing sent to the voice provider is the explanation text — no stems, no
report, no filename. That is the same reasoning as the reference-curve store:
the user's material stays on this machine, and a third party receives only a
paragraph the planner wrote about it.

ONE DEPENDENCY, ALREADY DECLARED. `httpx` is in `requirements.txt` for the API
tests, so the provider's REST endpoint is called directly rather than through
its SDK — an SDK pulls in a licence review for a few dozen transitive packages
in exchange for wrapping one POST.

THE KEY NEVER LEAVES THE ENVIRONMENT. `ELEVENLABS_API_KEY` is read from
`os.environ` (which `load_env` fills from the gitignored `.env.local`), sent as
a header, and never logged, echoed in an error, or written to disk.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

API_BASE = "https://api.elevenlabs.io/v1"

# One of the provider's premade voices ("Rachel"), chosen because it is present
# on every account tier and so works before the user has picked anything.
# Override with ELEVENLABS_VOICE_ID.
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# The multilingual model rather than the low-latency one: the explanation is
# read once per chain, so a second of extra latency costs nothing, and the
# text is full of units ("dBTP", "LUFS", "kHz") that the smaller model reads
# less reliably. Override with ELEVENLABS_MODEL_ID.
DEFAULT_MODEL_ID = "eleven_multilingual_v2"

# MP3 rather than PCM. This is a voice-over for a paragraph, not audio that
# enters the chain, so the codec is irrelevant to any measurement — and PCM
# output is gated to higher account tiers, which would make the feature depend
# on the billing plan.
OUTPUT_FORMAT = "mp3_44100_128"

# The explanation's own cap, mirrored here so a longer string is refused rather
# than billed. Keep in step with `GraphExplanation` in schemas/graph.py.
MAX_CHARS = 2400


class VoiceError(RuntimeError):
    """The voice could not be produced. The message is safe to show a user."""


def api_key() -> str | None:
    return os.environ.get("ELEVENLABS_API_KEY") or None


@dataclass
class Spoken:
    audio: bytes
    characters: int
    elapsed_s: float
    voice_id: str
    model_id: str
    # The provider's request id, for a support conversation. Empty if the
    # header was absent — never a fabricated value.
    request_id: str


def speak(text: str, *, voice_id: str | None = None, model_id: str | None = None,
          timeout_s: float = 60.0) -> Spoken:
    """One paragraph in, one MP3 out.

    Raises VoiceError with a user-safe message. The provider's error body is
    included when it is short, because "422" alone sends the user to the docs
    for something the body already names; the key is never in any message.
    """
    key = api_key()
    if not key:
        raise VoiceError("no ELEVENLABS_API_KEY; the explanation is on screen")
    text = text.strip()
    if not text:
        raise VoiceError("nothing to say: the explanation is empty")
    if len(text) > MAX_CHARS:
        raise VoiceError(f"the explanation is {len(text)} characters; the cap "
                         f"is {MAX_CHARS}")

    voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE_ID
    model_id = model_id or os.environ.get("ELEVENLABS_MODEL_ID") or DEFAULT_MODEL_ID

    started = time.monotonic()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(
                f"{API_BASE}/text-to-speech/{voice_id}",
                params={"output_format": OUTPUT_FORMAT},
                headers={"xi-api-key": key, "accept": "audio/mpeg"},
                json={"text": text, "model_id": model_id},
            )
    except httpx.HTTPError as error:
        raise VoiceError(f"could not reach ElevenLabs: {type(error).__name__}")
    elapsed = time.monotonic() - started

    if response.status_code != 200:
        body = response.text.strip()
        detail = body[:200] if body and len(body) <= 600 else ""
        raise VoiceError(f"ElevenLabs answered {response.status_code}"
                         + (f": {detail}" if detail else ""))

    return Spoken(
        audio=response.content,
        characters=len(text),
        elapsed_s=round(elapsed, 2),
        voice_id=voice_id,
        model_id=model_id,
        request_id=response.headers.get("request-id", ""),
    )
