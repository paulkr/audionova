"""Bridge to the C++ render engine.

Shells out to ``engine/build/anv_render`` rather than binding it. pybind11 is
in the plan and will happen, but not yet: a subprocess boundary costs a few
milliseconds of process start against a render measured in seconds, and it buys
the property that the Python side exercises EXACTLY the binary that ships.
An in-process binding is a second build configuration, and a second build
configuration is a second set of numerical results.

The other reason is blunter. The engine is where the audio quality lives and
Python is where the mistakes will be; a hard process boundary means a bug on
this side cannot corrupt a render, only fail to produce one.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.audio import wavio

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BINARY = os.path.join(_HERE, "engine", "build", "anv_render")


class RenderError(RuntimeError):
    """The engine refused to render, and said why."""


@dataclass
class Recipe:
    """The render description, mirroring engine/include/anv/io/Recipe.h.

    Two node lists, not one. ``nodes`` runs at the internal rate; ``master``
    runs after the output sample-rate conversion. The engine enforces which
    node types belong in which, so this class does not need to — and should
    not, because a second opinion about that rule is a way for the two to
    disagree.
    """

    nodes: list[dict[str, Any]] = field(default_factory=list)
    master: list[dict[str, Any]] = field(default_factory=list)
    internal_sample_rate: float = 96000.0
    output_sample_rate: float = 0.0  # 0 = keep the input's rate
    output_format: str = "float64"   # analysis default: no quantisation noise
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_version": "0.1.0",
            "internal_sample_rate": self.internal_sample_rate,
            "seed": self.seed,
            "output": {
                "sample_rate": self.output_sample_rate,
                "format": self.output_format,
            },
            "nodes": self.nodes,
            "master": self.master,
        }

    def add(self, node_type: str, node_id: str | None = None,
            mix: float = 1.0, bypass: bool = False, **params: Any) -> "Recipe":
        self.nodes.append(_node(node_type, node_id, mix, bypass, params))
        return self

    def add_master(self, node_type: str, node_id: str | None = None,
                   mix: float = 1.0, bypass: bool = False, **params: Any) -> "Recipe":
        self.master.append(_node(node_type, node_id, mix, bypass, params))
        return self


def _node(node_type: str, node_id: str | None, mix: float, bypass: bool,
          params: dict[str, Any]) -> dict[str, Any]:
    node: dict[str, Any] = {"type": node_type}
    if node_id is not None:
        node["id"] = node_id
    if mix != 1.0:
        node["mix"] = mix
    if bypass:
        node["bypass"] = True
    if params:
        node["params"] = params
    return node


def eq_band(freq_hz: float, gain_db: float, q: float = 1.0, band_type: int = 0) -> dict[str, float]:
    """One parametric EQ band. Type 0 is a bell; see ParametricEQ.h for the rest."""
    return {"type": band_type, "freq_hz": freq_hz, "gain_db": gain_db, "q": q}


def render(
    source: str | np.ndarray,
    recipe: Recipe | dict[str, Any],
    *,
    sample_rate: float | None = None,
    binary: str = DEFAULT_BINARY,
    block_frames: int | None = None,
    timeout_s: float = 600.0,
) -> wavio.AudioFile:
    """Render a file or an array through a recipe and read the result back.

    Arrays are written to a temporary 64-bit float WAV first. Float64 rather
    than the eventual delivery format, because anything narrower would put
    quantisation noise into a measurement of the engine's own behaviour and
    make a null test impossible.
    """
    if not os.path.exists(binary):
        raise RenderError(
            f"engine binary not found at {binary} — run `make render` in engine/"
        )

    recipe_dict = recipe.to_dict() if isinstance(recipe, Recipe) else recipe

    with tempfile.TemporaryDirectory(prefix="anv_render_") as work:
        if isinstance(source, str):
            in_path = source
        else:
            if sample_rate is None:
                raise RenderError("sample_rate is required when rendering an array")
            in_path = os.path.join(work, "in.wav")
            wavio.write_wav(in_path, source, sample_rate, 64, is_float=True)

        recipe_path = os.path.join(work, "recipe.json")
        out_path = os.path.join(work, "out.wav")
        with open(recipe_path, "w") as handle:
            json.dump(recipe_dict, handle)

        command = [binary, "--in", in_path, "--out", out_path,
                   "--recipe", recipe_path]
        if block_frames is not None:
            command += ["--block", str(block_frames)]

        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s
        )
        if result.returncode != 0:
            # The engine's own message is the useful one — it names the node
            # and the rule. Wrapping it in something vaguer would lose that.
            raise RenderError(result.stderr.strip() or f"exit {result.returncode}")

        return wavio.read_wav(out_path)


def render_array(
    samples: np.ndarray, sample_rate: float, recipe: Recipe | dict[str, Any], **kwargs: Any
) -> np.ndarray:
    """Render and return just the samples, trimmed to the input length.

    TRIMMED, and this is a choice rather than a convenience. A chain with a
    reverb produces output longer than its input, which is correct for a
    render and wrong for a comparison: a null test or a band-level delta needs
    two arrays of the same length describing the same moments. Callers wanting
    the tail should use ``render``.
    """
    out = render(samples, recipe, sample_rate=sample_rate, **kwargs)
    n = np.atleast_2d(samples).shape[1]
    return out.samples[:, :n]
