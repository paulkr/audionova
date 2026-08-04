"""A seeded, reproducible corpus of known chains for the recovery harness.

WHY GENERATE AT ALL, when `chains.CHAIN_TEMPLATES` is hand-written and the
docstring there argues for hand-writing. That argument was about NODE SETS: a
random permutation of node types produces chains nobody would build, and
scoring recovery on those measures nothing. It still holds, and the generator
respects it — the type sets here are the ones an engineer reaches for, and the
parameters are drawn from the ranges the sensitivity catalogue already swept.

What hand-writing cannot give is COVERAGE. Six templates cannot vary chain
length, parameter extremity and ordering independently, and with six of them a
single template moving by one node changes the headline by four points. The
generator varies those three axes deliberately and records what it did.

THREE THINGS IT ADDS THAT THE TEMPLATES COULD NOT.

  1. DETECTABILITY IS DECLARED, NOT ASSUMED. A 0.2 dB shelf is not recoverable
     by anyone. Scoring it as a miss produces a number that cannot improve, and
     a harness whose ceiling is below 1.0 for reasons nobody can fix is a
     harness people stop reading. Every generated node carries the tier it was
     drawn from, and the headline score counts only the tiers the sensitivity
     sweep actually demonstrated. The rest are counted and reported separately.

  2. THE FLOORS COME FROM THE SWEEP, NOT FROM INTUITION. `evidence_floor` reads
     `corpus.PERTURBATIONS` and returns the SMALLEST VALUE EVER SWEPT for that
     move. That is the smallest amount anything in this project has evidence is
     detectable. Nothing below it has been shown to be recoverable, so nothing
     below it is scored as though it should have been. Deriving the number
     rather than writing it down also means it cannot drift from the catalogue
     it claims to come from.

  3. ORDERING IS BALANCED BY CONSTRUCTION. The `ordering_pair` family emits
     every chain twice, the second time exactly reversed. A predictor that
     emits a fixed sequence scores +rho on one and -rho on the other, so its
     average is zero and any positive score is ordering information rather than
     a property of the corpus. That is the specific defect recorded in
     findings.md under "An ordering metric that measures the corpus".

DETERMINISM. Everything is drawn from `random.Random(seed)` through `_uniform`
and `_pick`, which use only `Random.random()` — the one method CPython
guarantees reproduces across versions. The result is digested and committed as
`manifests/recovery_v1.json`, so a change in the headline score is attributable
to the planner rather than to new random chains. `manifest_digest` mismatching
the committed file is a HARD ERROR, never a silent regeneration: a corpus that
quietly reshapes itself is indistinguishable from a planner that got worse.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.eval import corpus

GENERATOR_VERSION = "recovery_v1"

# The date this corpus was fixed. Any constant would do; what matters is that
# it is written down and never changed without regenerating the manifest.
DEFAULT_SEED = 20260804

MANIFEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifests")
DEFAULT_MANIFEST = os.path.join(MANIFEST_DIR, f"{GENERATOR_VERSION}.json")

# How far each tier sits from the smallest amount the sensitivity sweep
# demonstrated. Multipliers, not absolute values, so one table covers dB, Hz,
# ratio and mix without any of them being written twice.
#
#   strong / demonstrated  the sweep showed these are readable off the report
#   borderline             between the sweep's floor and nothing; NO EVIDENCE
#                          either way, so scoring it either way would be an
#                          invention
#   sub_threshold          a twentieth of the smallest demonstrated amount. A
#                          0.2 dB shelf. Nobody recovers this, and a harness
#                          that punishes a planner for not recovering it has
#                          built itself a ceiling below 1.0
TIER_SCALE: dict[str, tuple[float, float]] = {
    "strong": (2.0, 3.0),
    "demonstrated": (1.0, 2.0),
    "borderline": (0.35, 0.7),
    "sub_threshold": (0.03, 0.10),
}

# Only these two count toward the headline. The other two are reported beside
# it, never folded into it.
IN_EVIDENCE_TIERS = ("strong", "demonstrated")


def _uniform(rng: random.Random, low: float, high: float) -> float:
    return low + (high - low) * rng.random()


def _pick(rng: random.Random, options):
    return options[min(int(rng.random() * len(options)), len(options) - 1)]


def _shuffled(rng: random.Random, items: list) -> list:
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = min(int(rng.random() * (i + 1)), i)
        out[i], out[j] = out[j], out[i]
    return out


def _sign(rng: random.Random) -> float:
    return 1.0 if rng.random() < 0.5 else -1.0


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    import math

    return math.exp(_uniform(rng, math.log(low), math.log(high)))


def evidence_floor(perturbation_name: str, *, relative_to: float = 0.0) -> float:
    """The smallest amount of this move the sensitivity sweep ever applied.

    READ FROM THE CATALOGUE rather than written down here, so the two cannot
    disagree. `relative_to` is for parameters whose "amount" is a distance from
    a neutral value rather than the value itself — a stereo width of 100% is no
    change at all, so the amount is |width - 100|, and a compression ratio of
    1:1 is no change, so the amount is ratio - 1.

    UNKNOWN NAMES ARE A HARD ERROR. A typo silently returning a default would
    put an invented floor into the scoring, which is the one thing this function
    exists to prevent.
    """
    if perturbation_name not in corpus.PERTURBATIONS_BY_NAME:
        raise KeyError(
            f"{perturbation_name!r} is not in the sensitivity catalogue, so "
            "there is no evidence for any detection floor for it")
    values = corpus.PERTURBATIONS_BY_NAME[perturbation_name].values
    amounts = [abs(v - relative_to) for v in values]
    positive = [a for a in amounts if a > 0.0]
    if not positive:
        raise ValueError(f"{perturbation_name!r} sweeps no non-zero amount")
    return min(positive)


def _amount(rng: random.Random, tier: str, floor: float) -> float:
    if tier not in TIER_SCALE:
        raise KeyError(f"unknown tier {tier!r}")
    low, high = TIER_SCALE[tier]
    return _uniform(rng, low * floor, high * floor)


# ---------------------------------------------------------------------------
# The move catalogue
#
# A "move" is one thing an engineer does. It is not the same as a node type:
# four of these are `parametric_eq`, and the recovery score is over node types,
# but the parameter score needs to know which band was meant.


@dataclass(frozen=True)
class Move:
    name: str
    node_type: str
    # Which entry in the sensitivity catalogue backs this move's detection
    # floor. Named rather than numeric so the link is checkable.
    evidence: str
    build: Callable[[random.Random, str], dict[str, Any]]
    neutral: float = 0.0
    needs_stereo: bool = False

    @property
    def floor(self) -> float:
        return evidence_floor(self.evidence, relative_to=self.neutral)


def _round(value: float, places: int = 4) -> float:
    # Rounded at generation time so the manifest digest is stable against
    # float-repr differences, and so the recorded ground truth is the number
    # the engine is actually handed.
    return float(round(value, places))


def _bell(rng: random.Random, tier: str) -> dict[str, Any]:
    floor = evidence_floor("mid_bell")
    gain = _sign(rng) * _amount(rng, tier, floor)
    return {"bands": [{
        "filter_type": "bell",
        # Log-uniform and OFF the third-octave grid on purpose. A ground truth
        # sitting exactly on a band centre would flatter any recogniser that
        # reports band centres, which is what the frequency error is measuring.
        "center_freq_hz": _round(_log_uniform(rng, 90.0, 9000.0), 1),
        "gain_db": _round(gain, 2),
        "q_factor": _pick(rng, (0.7, 1.0, 1.4, 2.0, 3.0)),
    }]}


def _high_shelf(rng: random.Random, tier: str) -> dict[str, Any]:
    floor = evidence_floor("high_shelf")
    return {"bands": [{
        "filter_type": "highshelf",
        "center_freq_hz": _round(_log_uniform(rng, 2500.0, 11000.0), 1),
        "gain_db": _round(_sign(rng) * _amount(rng, tier, floor), 2),
        "q_factor": 0.707,
    }]}


def _low_shelf(rng: random.Random, tier: str) -> dict[str, Any]:
    floor = evidence_floor("low_shelf")
    return {"bands": [{
        "filter_type": "lowshelf",
        "center_freq_hz": _round(_log_uniform(rng, 70.0, 400.0), 1),
        "gain_db": _round(_sign(rng) * _amount(rng, tier, floor), 2),
        "q_factor": 0.707,
    }]}


def _highpass(rng: random.Random, tier: str) -> dict[str, Any]:
    # The "amount" of a high-pass is its corner: the higher it sits the more it
    # removes. The schema floor is 20 Hz, which is below essentially every
    # source's content, so the sub-threshold tier lands there and correctly
    # removes nothing.
    floor = evidence_floor("highpass")
    corner = min(max(_amount(rng, tier, floor), 20.0), 200.0)
    return {"bands": [{
        "filter_type": "highpass",
        "center_freq_hz": _round(corner, 1),
        "gain_db": 0.0,
        "q_factor": 0.707,
    }]}


def _compressor(rng: random.Random, tier: str) -> dict[str, Any]:
    # Amount is ratio - 1, because 1:1 is the no-op. Scaling the ratio itself
    # would make the sub-threshold tier ask for 0.06:1, which is not a weak
    # compressor, it is an illegal one.
    floor = evidence_floor("compression", relative_to=1.0)
    ratio = 1.0 + _amount(rng, tier, floor)
    return {
        "ratio_to_1": _round(min(ratio, 20.0), 3),
        "knee_db": _pick(rng, (0.0, 3.0, 6.0, 9.0)),
        "attack_ms": _round(_log_uniform(rng, 3.0, 40.0), 1),
        "release_ms": _round(_log_uniform(rng, 60.0, 400.0), 0),
        # Filled in per source at build time, since a fixed threshold makes the
        # same chain mean different things on a bass and an acoustic guitar.
        "threshold_offset_db": -6.0,
    }


def _drive(rng: random.Random, tier: str) -> dict[str, Any]:
    floor = evidence_floor("drive")
    amount = min(_amount(rng, tier, floor), 30.0)
    return {"drive_db": _round(amount, 2),
            "output_trim_db": _round(-amount * 0.5, 2)}


def _plate(rng: random.Random, tier: str) -> dict[str, Any]:
    floor = evidence_floor("plate_reverb")
    return {"__mix__": _round(min(_amount(rng, tier, floor), 1.0), 4),
            "decay_s": _round(_uniform(rng, 0.8, 3.0), 2),
            "pre_delay_ms": _round(_uniform(rng, 0.0, 40.0), 1),
            "damping_hz": _round(_uniform(rng, 4000.0, 9000.0), 0)}


def _delay(rng: random.Random, tier: str) -> dict[str, Any]:
    floor = evidence_floor("delay")
    return {"__mix__": _round(min(_amount(rng, tier, floor), 1.0), 4),
            "time_ms": _round(_uniform(rng, 120.0, 450.0), 0),
            "feedback_percent": _round(_uniform(rng, 10.0, 55.0), 1)}


def _width(rng: random.Random, tier: str) -> dict[str, Any]:
    floor = evidence_floor("stereo_width", relative_to=100.0)
    amount = _amount(rng, tier, floor)
    percent = 100.0 + _sign(rng) * amount
    return {"width_percent": _round(min(max(percent, 0.0), 250.0), 2)}


MOVES: tuple[Move, ...] = (
    Move("bell", "parametric_eq", "mid_bell", _bell),
    Move("high_shelf", "parametric_eq", "high_shelf", _high_shelf),
    Move("low_shelf", "parametric_eq", "low_shelf", _low_shelf),
    Move("highpass", "parametric_eq", "highpass", _highpass),
    Move("compression", "compressor", "compression", _compressor),
    Move("drive", "drive", "drive", _drive),
    Move("plate_reverb", "plate_reverb", "plate_reverb", _plate),
    Move("delay", "delay", "delay", _delay),
    Move("stereo_width", "stereo_width", "stereo_width", _width,
         neutral=100.0, needs_stereo=True),
)

MOVES_BY_NAME = {m.name: m for m in MOVES}

# For the ordering family only. Every member is a DIFFERENT node type, because
# the ordering score is computed over the first occurrence of each type shared
# between prediction and truth: reversing "eq, comp, eq" gives "eq, comp, eq"
# back, so a mirrored pair built from repeated types is not a mirrored pair at
# all and would quietly unbalance the very thing the family exists to balance.
ORDERING_MOVES = ("bell", "compression", "drive", "plate_reverb", "delay")


# ---------------------------------------------------------------------------
# Chain specifications


@dataclass
class NodeSpec:
    """One node of a generated chain, with its tier recorded."""

    move: str
    node_type: str
    tier: str
    params: dict[str, Any]
    mix: float = 1.0

    @property
    def in_evidence(self) -> bool:
        return self.tier in IN_EVIDENCE_TIERS

    def to_dict(self) -> dict[str, Any]:
        return {"move": self.move, "node_type": self.node_type, "tier": self.tier,
                "params": self.params, "mix": self.mix}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeSpec":
        return cls(move=data["move"], node_type=data["node_type"],
                   tier=data["tier"], params=dict(data["params"]),
                   mix=float(data["mix"]))


@dataclass
class ChainSpec:
    name: str
    family: str
    nodes: list[NodeSpec] = field(default_factory=list)
    # Only chains whose ground-truth order has an exact mirror in the corpus.
    # Everything else is excluded from the ordering metric, because a corpus
    # that leans conventional hands a fixed-sequence predictor a free positive
    # score — the defect this whole family exists to remove.
    ordering_eligible: bool = False

    @property
    def node_types(self) -> list[str]:
        return [n.node_type for n in self.nodes]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "family": self.family,
                "ordering_eligible": self.ordering_eligible,
                "nodes": [n.to_dict() for n in self.nodes]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChainSpec":
        return cls(name=data["name"], family=data["family"],
                   ordering_eligible=bool(data["ordering_eligible"]),
                   nodes=[NodeSpec.from_dict(n) for n in data["nodes"]])


def _make_node(rng: random.Random, move: Move, tier: str) -> NodeSpec:
    params = move.build(rng, tier)
    mix = float(params.pop("__mix__", 1.0))
    return NodeSpec(move=move.name, node_type=move.node_type, tier=tier,
                    params=params, mix=mix)


# The order an engineer would place these in. Used only to give the non-ordering
# families a plausible sequence; the ordering family deliberately ignores it.
_PLAUSIBLE_ORDER = ("highpass", "bell", "low_shelf", "high_shelf", "compression",
                    "drive", "stereo_width", "delay", "plate_reverb")


def _in_plausible_order(names: list[str]) -> list[str]:
    return sorted(names, key=_PLAUSIBLE_ORDER.index)


def generate(seed: int = DEFAULT_SEED) -> list[ChainSpec]:
    """The corpus. Deterministic in `seed` and in nothing else.

    Four families, each answering a question the others cannot:

      ordering_pair   can order be read at all? Balanced, so a fixed-sequence
                      predictor scores exactly zero.
      varied          length and node-type coverage, duplicates included,
                      because "EQ, dynamics, EQ" is the commonest structure
                      there is and a set-based recogniser must not get it free.
      faint           at least one node below the sweep's floor. Excluded from
                      the headline, counted on its own.
      single          one node, one tier. The easy case, kept because a
                      recogniser failing here has a bug rather than a limit.
    """
    rng = random.Random(seed)
    specs: list[ChainSpec] = []

    # --- ordering pairs -----------------------------------------------------
    for index in range(12):
        length = 2 if index % 3 == 0 else 3
        chosen = _shuffled(rng, list(ORDERING_MOVES))[:length]
        tier = "strong" if index % 2 == 0 else "demonstrated"
        nodes = [_make_node(rng, MOVES_BY_NAME[name], tier) for name in chosen]

        specs.append(ChainSpec(name=f"pair{index:02d}_fwd", family="ordering_pair",
                               nodes=nodes, ordering_eligible=True))
        # THE EXACT REVERSE, sharing the same parameter draws. Reversing the
        # nodes rather than redrawing them means the pair differs in ordering
        # and in nothing else, so the two cases cannot be separated by anything
        # except the sequence.
        specs.append(ChainSpec(name=f"pair{index:02d}_rev", family="ordering_pair",
                               nodes=list(reversed(nodes)), ordering_eligible=True))

    # --- varied ------------------------------------------------------------
    pool = [m.name for m in MOVES]
    for index in range(20):
        length = 1 + (index % 5)
        names = [_pick(rng, pool) for _ in range(length)]
        tier = _pick(rng, ("strong", "demonstrated", "demonstrated"))
        ordered = _in_plausible_order(names)
        nodes = [_make_node(rng, MOVES_BY_NAME[name], tier) for name in ordered]
        specs.append(ChainSpec(name=f"varied{index:02d}", family="varied",
                               nodes=nodes))

    # --- faint --------------------------------------------------------------
    #
    # DELIBERATELY UNRECOVERABLE, and the point of including them is that they
    # are scored separately rather than counted as misses. A harness that marks
    # a planner down for not hearing a 0.2 dB shelf has a ceiling below 1.0 for
    # a reason no planner can address, and a metric nobody can move is a metric
    # nobody reads.
    for index in range(14):
        faint_tier = "sub_threshold" if index % 2 == 0 else "borderline"
        faint = _make_node(rng, MOVES_BY_NAME[_pick(rng, pool)], faint_tier)
        nodes = [faint]
        if index % 3 != 0:
            # Mixed: one faint node alongside a clearly detectable one, which is
            # the case that separates "found nothing" from "found what was
            # there and correctly stopped".
            loud = _make_node(rng, MOVES_BY_NAME[_pick(rng, pool)], "strong")
            nodes = [n for _, n in sorted(
                ((_PLAUSIBLE_ORDER.index(n.move), n) for n in (faint, loud)),
                key=lambda pair: pair[0])]
        specs.append(ChainSpec(name=f"faint{index:02d}", family="faint", nodes=nodes))

    # --- singles ------------------------------------------------------------
    for index, move in enumerate(MOVES):
        tier = "strong" if index % 2 == 0 else "demonstrated"
        specs.append(ChainSpec(name=f"single_{move.name}", family="single",
                               nodes=[_make_node(rng, move, tier)]))

    return specs


# ---------------------------------------------------------------------------
# Manifest


def manifest_body(specs: list[ChainSpec], seed: int) -> dict[str, Any]:
    return {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "chain_count": len(specs),
        "chains": [s.to_dict() for s in specs],
    }


def digest(specs: list[ChainSpec]) -> str:
    """A hash of the chains alone, so the manifest can carry its own checksum."""
    payload = json.dumps([s.to_dict() for s in specs], sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_manifest(path: str = DEFAULT_MANIFEST, seed: int = DEFAULT_SEED) -> str:
    specs = generate(seed)
    body = manifest_body(specs, seed)
    body["digest"] = digest(specs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(body, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return body["digest"]


class ManifestMismatch(RuntimeError):
    """The committed corpus and the generator disagree.

    A HARD ERROR rather than a regeneration. The manifest exists so that a
    change in the headline score is attributable to the planner; a corpus that
    silently reshapes itself when the generator changes is indistinguishable
    from a planner that got worse, which is the whole failure it was added to
    prevent.
    """


def load_manifest(path: str = DEFAULT_MANIFEST) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def verify_manifest(path: str = DEFAULT_MANIFEST) -> list[ChainSpec]:
    """Regenerate from the recorded seed and check it against the committed file."""
    body = load_manifest(path)
    if body.get("generator_version") != GENERATOR_VERSION:
        raise ManifestMismatch(
            f"manifest is {body.get('generator_version')!r} but the generator is "
            f"{GENERATOR_VERSION!r}")

    committed = [ChainSpec.from_dict(c) for c in body["chains"]]
    committed_digest = digest(committed)
    if committed_digest != body.get("digest"):
        raise ManifestMismatch(
            "the manifest's own digest does not match its contents: "
            f"{body.get('digest')} vs {committed_digest}")

    regenerated = generate(int(body["seed"]))
    if digest(regenerated) != committed_digest:
        raise ManifestMismatch(
            f"regenerating from seed {body['seed']} produced a different corpus "
            f"({digest(regenerated)}) than the committed manifest "
            f"({committed_digest}). Rerun `python -m backend.eval.generate` and "
            "read the diff before accepting it — every score in the report moved.")
    return committed


if __name__ == "__main__":
    written = write_manifest()
    print(f"{DEFAULT_MANIFEST}\n{written}")
