"""Blinded, seeded, randomised listening sessions.

WHAT BLINDING HAS TO MEAN HERE. Not "the labels are hidden in the interface" —
that is a rendering decision, and a rendering decision can be undone with a
right-click. It has to mean the session payload the interface receives contains
no information from which the mapping could be reconstructed: no condition
names, no item names, no source paths, no repeat flags, no ordering that
correlates with condition, and no clip identifier that is a function of what
the clip is. The mapping lives in a separate key file that the interface never
sees and the server never serves.

THE PLAN IS SEPARATE FROM THE MATERIALISATION, and this is deliberate. Every
property worth testing about a blinded session — that the mapping is not
recoverable, that the randomisation is seeded and reproducible across
processes, that repeats are placed apart, that A/B assignment is balanced — is
a property of the plan and needs no audio at all. Testing them through the
audio path would make them slow, which makes them run rarely, which is the same
as not having them.

CLIP IDENTIFIERS ARE PER TRIAL OCCURRENCE, NOT PER FILE. A repeat trial that
reused the same identifiers would announce itself, and a repeat trial that
announces itself measures whether a listener can remember, not whether they are
consistent. Two occurrences of the same pair therefore carry four distinct
identifiers, and the server resolves them privately.

WHAT THIS DOES NOT HIDE, stated because a blinding claim with unwritten limits
gets trusted past them:

  * The request text is shown, because one of the two scales is "does it match
    the request?" and the listener cannot answer it otherwise. Two trials
    carrying the same request are probably a repeat, and an attentive listener
    may notice. Repeats are placed at least `min_repeat_separation` trials
    apart so the pair is never adjacent, and A/B is re-randomised between the
    two occurrences, so noticing a repeat does not tell a listener what they
    said last time.
  * Two clip identifiers that resolve to the same file are served as
    byte-identical responses. A listener determined to defeat the blinding
    could compare downloads. This is a panel of colleagues, not an adversary,
    and padding the files to hide it would mean shipping audio nobody measured.

See `docs/open-questions.md`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from backend.listen import match

# How far apart the two occurrences of a repeated pair must sit.
MIN_REPEAT_SEPARATION = 3

# Give up rather than emit an order that violates the separation. A session
# that quietly placed a repeat adjacent to its original would produce a
# consistency number measuring short-term memory.
_MAX_SHUFFLE_ATTEMPTS = 400

# The two scales. They are collected separately because they dissociate: a
# render can do exactly what was asked for and sound amateur, and it can sound
# beautiful while ignoring the request. Averaging them into one number loses
# the only distinction that tells you which half is broken.
SCALES: tuple[dict[str, Any], ...] = (
    {
        "key": "match",
        "prompt": "Does it do what was asked for?",
        "min": 1,
        "max": 5,
        "labels": {
            "1": "not at all",
            "2": "barely",
            "3": "partly",
            "4": "mostly",
            "5": "exactly",
        },
    },
    {
        "key": "quality",
        "prompt": "Does it sound professional?",
        "min": 1,
        "max": 5,
        "labels": {
            "1": "unusable",
            "2": "amateur",
            "3": "acceptable",
            "4": "good",
            "5": "release-ready",
        },
    },
)

SCALE_KEYS = tuple(scale["key"] for scale in SCALES)
PREFERENCES = ("A", "B", "tie")


class SessionError(ValueError):
    """The session as described cannot be built."""


@dataclass(frozen=True)
class Comparison:
    """One (source, processed-A, processed-B) triple and the request behind it.

    `conditions` is an ordered pair of (name, path). The order it is written in
    must not survive into the session — see `test_listen.py`, which builds the
    same session with the pair written both ways round and requires the two
    payloads to be identical.
    """

    item: str
    request: str
    source: str
    conditions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.item:
            raise SessionError("a comparison needs an item name")
        if not self.request:
            raise SessionError(
                f"{self.item}: a comparison needs the request text, because "
                "one of the two scales asks whether the render matched it")
        if len(self.conditions) != 2:
            raise SessionError(
                f"{self.item}: an A/B trial compares exactly two conditions, "
                f"got {len(self.conditions)}")
        names = [name for name, _ in self.conditions]
        if len(set(names)) != 2:
            raise SessionError(f"{self.item}: both conditions are named {names[0]!r}")
        for name, path in self.conditions:
            if not name:
                raise SessionError(f"{self.item}: a condition needs a name")
            if not path:
                raise SessionError(f"{self.item}: condition {name!r} has no audio")

    @property
    def condition_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.conditions)

    def path_for(self, condition: str) -> str:
        for name, path in self.conditions:
            if name == condition:
                return path
        raise SessionError(f"{self.item}: no condition named {condition!r}")


def comparison(item: str, request: str, source: str,
               conditions: Sequence[tuple[str, str]] | dict[str, str]) -> Comparison:
    """Build a Comparison from either an ordered sequence or a mapping."""
    if isinstance(conditions, dict):
        pairs = tuple(conditions.items())
    else:
        pairs = tuple((name, path) for name, path in conditions)
    return Comparison(item=item, request=request, source=source, conditions=pairs)


@dataclass(frozen=True)
class Trial:
    """One presentation. `condition_a` / `condition_b` are KEY-SIDE ONLY."""

    trial_id: str
    listener: str
    position: int
    comparison_index: int
    condition_a: str
    condition_b: str
    clip_a: str
    clip_b: str
    reference_clip: str
    repeat_of: str | None


@dataclass(frozen=True)
class SessionPlan:
    session_id: str
    seed: int
    comparisons: tuple[Comparison, ...]
    listeners: tuple[str, ...]
    trials: tuple[Trial, ...]
    repeats: int
    min_repeat_separation: int

    def trials_for(self, listener: str) -> tuple[Trial, ...]:
        if listener not in self.listeners:
            raise SessionError(f"no listener {listener!r} in this session")
        return tuple(t for t in self.trials if t.listener == listener)

    def trial(self, trial_id: str) -> Trial:
        for candidate in self.trials:
            if candidate.trial_id == trial_id:
                return candidate
        raise SessionError(f"no trial {trial_id!r} in this session")

    # -- the two documents -------------------------------------------------

    def public_payload(self) -> dict[str, Any]:
        """What the interface is allowed to know.

        Everything here is either opaque or is something the listener is going
        to hear anyway. Nothing here names a condition, an item, a file, or a
        repeat.
        """
        listeners: dict[str, Any] = {}
        for listener in self.listeners:
            listeners[listener] = [
                {
                    "trial_id": trial.trial_id,
                    "request": self.comparisons[trial.comparison_index].request,
                    "clip_a": trial.clip_a,
                    "clip_b": trial.clip_b,
                    "reference": trial.reference_clip,
                }
                for trial in self.trials_for(listener)
            ]
        return {
            "session_id": self.session_id,
            "scales": [dict(scale) for scale in SCALES],
            "preferences": list(PREFERENCES),
            "listeners": listeners,
        }

    def key_payload(self) -> dict[str, Any]:
        """Everything, including the mapping. Never served."""
        return {
            "session_id": self.session_id,
            "seed": self.seed,
            "repeats": self.repeats,
            "min_repeat_separation": self.min_repeat_separation,
            "comparisons": [
                {
                    "index": index,
                    "item": item.item,
                    "request": item.request,
                    "source": item.source,
                    "conditions": {name: path for name, path in item.conditions},
                }
                for index, item in enumerate(self.comparisons)
            ],
            "trials": {
                trial.trial_id: {
                    "listener": trial.listener,
                    "position": trial.position,
                    "comparison": trial.comparison_index,
                    "item": self.comparisons[trial.comparison_index].item,
                    "condition_a": trial.condition_a,
                    "condition_b": trial.condition_b,
                    "clip_a": trial.clip_a,
                    "clip_b": trial.clip_b,
                    "reference": trial.reference_clip,
                    "repeat_of": trial.repeat_of,
                }
                for trial in self.trials
            },
        }


def _rng(seed: int, tag: str) -> random.Random:
    """A seeded generator that survives a process restart.

    blake2b rather than `hash()`: string hashing is salted per process, so a
    session derived from it would be reproducible within one run and different
    in the next, which is the worst of both — it looks deterministic right up
    until someone tries to rebuild a session from its recorded seed.
    """
    digest = hashlib.blake2b(f"{int(seed)}\x00{tag}".encode("utf-8"),
                             digest_size=32).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _opaque(rng: random.Random) -> str:
    return f"{rng.getrandbits(64):016x}"


# Public names for the two primitives, so the screen and the runner can derive
# their own streams from the same seed without reaching into this module.
derive_rng = _rng
opaque_id = _opaque


def _ordered_slots(rng: random.Random, slots: Sequence[int],
                   separation: int) -> list[int]:
    """Shuffle, rejecting any order that puts a repeat too close to its twin."""
    order = list(slots)
    for _ in range(_MAX_SHUFFLE_ATTEMPTS):
        rng.shuffle(order)
        seen: dict[int, int] = {}
        ok = True
        for position, value in enumerate(order):
            previous = seen.get(value)
            if previous is not None and position - previous < separation:
                ok = False
                break
            seen[value] = position
        if ok:
            return list(order)
    raise SessionError(
        f"could not place {len(slots) - len(set(slots))} repeat(s) at least "
        f"{separation} trials apart among {len(set(slots))} comparisons. "
        "Either add comparisons, cut repeats, or lower the separation — the "
        "one thing not on offer is an adjacent repeat, which measures memory "
        "rather than consistency.")


def plan_session(
    comparisons: Sequence[Comparison],
    listeners: Sequence[str],
    *,
    seed: int,
    repeats: int = 0,
    min_repeat_separation: int = MIN_REPEAT_SEPARATION,
    session_id: str | None = None,
) -> SessionPlan:
    """Decide the whole session. Pure: no audio, no filesystem, no clock."""
    comparisons = tuple(comparisons)
    if not comparisons:
        raise SessionError("a session needs at least one comparison")

    listeners = tuple(listeners)
    if not listeners:
        raise SessionError("a session needs at least one listener")
    if len(set(listeners)) != len(listeners):
        raise SessionError(f"listener ids must be unique, got {list(listeners)}")
    if any(not listener for listener in listeners):
        raise SessionError("every listener needs an id")

    if repeats < 0:
        raise SessionError(f"repeats must not be negative, got {repeats}")
    if repeats > len(comparisons):
        raise SessionError(
            f"asked for {repeats} repeat trials from {len(comparisons)} "
            "comparisons; a pair cannot be repeated more than once")
    if min_repeat_separation < 1:
        raise SessionError(
            f"min_repeat_separation must be at least 1, got "
            f"{min_repeat_separation}")

    # WHICH pairs repeat is a property of the session, not of a listener: every
    # listener answers the same repeated pairs, so their consistency numbers
    # are comparable with each other.
    repeated = sorted(_rng(seed, "repeats").sample(range(len(comparisons)), repeats))
    slots = list(range(len(comparisons))) + repeated

    if session_id is None:
        session_id = _opaque(_rng(seed, "session_id"))

    trials: list[Trial] = []
    for listener in listeners:
        rng = _rng(seed, f"listener:{listener}")
        order = _ordered_slots(rng, slots, min_repeat_separation)

        first_seen: dict[int, str] = {}
        for position, index in enumerate(order):
            # DRAW ORDER IS FIXED AND INDEPENDENT OF CONTENT. The side bit is
            # drawn before it is used to select a condition, and the three clip
            # ids follow regardless. That is what makes the payload identical
            # when the two conditions are written the other way round.
            flip = rng.getrandbits(1)
            trial_id = _opaque(rng)
            clip_a = _opaque(rng)
            clip_b = _opaque(rng)
            reference = _opaque(rng)

            names = comparisons[index].condition_names
            condition_a, condition_b = (names[1], names[0]) if flip else names

            trials.append(Trial(
                trial_id=trial_id,
                listener=listener,
                position=position,
                comparison_index=index,
                condition_a=condition_a,
                condition_b=condition_b,
                clip_a=clip_a,
                clip_b=clip_b,
                reference_clip=reference,
                repeat_of=first_seen.get(index),
            ))
            first_seen.setdefault(index, trial_id)

    ids = [trial.trial_id for trial in trials]
    clips = [clip for trial in trials
             for clip in (trial.clip_a, trial.clip_b, trial.reference_clip)]
    if len(set(ids)) != len(ids) or len(set(clips)) != len(clips):
        raise SessionError(
            "identifier collision while planning the session — rebuild with a "
            "different seed rather than shipping two trials that share an id")

    return SessionPlan(
        session_id=session_id,
        seed=int(seed),
        comparisons=comparisons,
        listeners=listeners,
        trials=tuple(trials),
        repeats=repeats,
        min_repeat_separation=min_repeat_separation,
    )


# --- materialisation ---------------------------------------------------------


@dataclass(frozen=True)
class BuiltSession:
    directory: str
    plan: SessionPlan
    session_json: str
    key_json: str
    clips_dir: str
    responses_dir: str
    groups: tuple[match.MatchedGroup, ...]
    clip_files: dict[str, str] = field(default_factory=dict)


def build_session(
    comparisons: Sequence[Comparison],
    listeners: Sequence[str],
    directory: str,
    *,
    seed: int,
    repeats: int = 0,
    min_repeat_separation: int = MIN_REPEAT_SEPARATION,
    session_id: str | None = None,
    gain_solver: match.GainSolver = match.match_to_quietest,
    tolerance_lu: float = match.TOLERANCE_LU,
    ceiling_dbtp: float = match.CEILING_DBTP,
    bit_depth: int = match.BIT_DEPTH,
) -> BuiltSession:
    """Plan the session, loudness-match every clip, and write it out.

    Raises before writing `session.json` if any comparison fails to match, so
    a directory containing a session is a directory whose audio was verified.
    """
    plan = plan_session(comparisons, listeners, seed=seed, repeats=repeats,
                        min_repeat_separation=min_repeat_separation,
                        session_id=session_id)

    clips_dir = os.path.join(directory, "clips")
    responses_dir = os.path.join(directory, "responses")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(responses_dir, exist_ok=True)

    store_rng = _rng(seed, "store")
    groups: list[match.MatchedGroup] = []
    # (comparison index, role) -> written file. Roles are "source" and the two
    # condition names; the file names themselves are opaque, so listing the
    # directory tells you nothing either.
    stored: dict[tuple[int, str], str] = {}

    for index, item in enumerate(plan.comparisons):
        roles = ["source"] + list(item.condition_names)
        inputs = [item.source] + [item.path_for(name) for name in item.condition_names]
        outputs = [os.path.join(clips_dir, f"{_opaque(store_rng)}.wav")
                   for _ in roles]
        group = match.match_group(
            inputs, outputs,
            gain_solver=gain_solver,
            tolerance_lu=tolerance_lu,
            ceiling_dbtp=ceiling_dbtp,
            bit_depth=bit_depth,
            what=f"comparison {index} ({item.item})",
        )
        groups.append(group)
        for role, path in zip(roles, outputs):
            stored[(index, role)] = path

    clip_files: dict[str, str] = {}
    for trial in plan.trials:
        clip_files[trial.clip_a] = stored[(trial.comparison_index, trial.condition_a)]
        clip_files[trial.clip_b] = stored[(trial.comparison_index, trial.condition_b)]
        clip_files[trial.reference_clip] = stored[(trial.comparison_index, "source")]

    key = plan.key_payload()
    key["clips"] = {clip_id: os.path.relpath(path, directory)
                    for clip_id, path in clip_files.items()}
    key["loudness"] = {
        os.path.relpath(clip.path, directory): {
            "lufs": round(clip.lufs, 4),
            "true_peak_dbtp": round(clip.true_peak_dbtp, 4),
            "gain_db": round(clip.gain_db, 4),
        }
        for group in groups for clip in group.clips
    }
    key["match"] = {
        "tolerance_lu": tolerance_lu,
        "ceiling_dbtp": ceiling_dbtp,
        "verified_spread_lu": [round(group.spread_lu, 4) for group in groups],
    }

    session_json = os.path.join(directory, "session.json")
    key_json = os.path.join(directory, "key.json")
    with open(session_json, "w") as handle:
        json.dump(plan.public_payload(), handle, indent=2, sort_keys=True)
    with open(key_json, "w") as handle:
        json.dump(key, handle, indent=2, sort_keys=True)

    return BuiltSession(
        directory=directory,
        plan=plan,
        session_json=session_json,
        key_json=key_json,
        clips_dir=clips_dir,
        responses_dir=responses_dir,
        groups=tuple(groups),
        clip_files=clip_files,
    )


def load_key(path: str) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def leaks(payload: Any, secrets: Iterable[str]) -> list[str]:
    """Which of `secrets` appear anywhere in `payload` once serialised.

    Used by the blinding test and by `run_session` before it writes anything.
    Substring matching on the serialised document, deliberately: a condition
    name embedded in an identifier is exactly the leak worth catching, and a
    structural walk that only inspected values would miss it in a key.
    """
    text = json.dumps(payload, sort_keys=True).lower()
    return sorted({secret for secret in secrets
                   if secret and str(secret).lower() in text})
