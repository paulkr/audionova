"""Prompt assembly, and the cache boundary.

THE ECONOMICS ARE INVERTED FROM THE OBVIOUS ASSUMPTION. DSP compute is about
$0.0007 for a three-minute render. A frontier planning turn is $0.15 to $0.50 —
the model costs two to seven hundred times the audio processing. So the thing
worth engineering is the prompt, not the renderer.

The prefix is large and static: role framing, the annotated schema, the DSP
source of every effect, the glossary, the preset index. Measured against the
live API it is **30,938 tokens** that never change between requests, so they are
cached. The volatile part — the measurement report, the classification, the
user's sentence — sits after the boundary.

That is larger than the plan's ~18.5 k budget, and the discrepancy was a
counting error rather than scope creep: the prefix is mostly C++ source and JSON
schema, which tokenise far denser than the prose a 4-characters-per-token rule
assumes.

ONE TIMESTAMP IN THE PREFIX DESTROYS THE ENTIRE SAVING. A request id, a user id,
a formatted date, anything per-request in the system block moves the cache
boundary to before it and every subsequent token becomes a cache miss. This is
silent: the request succeeds, the answer is fine, and the bill is five times
larger. So the prefix builders take NO arguments, and a test asserts that two
independently-built prefixes are byte-identical.

THE TWO STAGES DO NOT SHARE A PREFIX, AND SHARING ONE WAS A BUG. They ran the
same 30,680-token prefix and used different models — Opus to reason, Sonnet to
transcribe — so the prefix was written into two separate caches and neither call
could ever read the other's. The cost was paid twice for a saving that could
only ever be collected once.

Splitting them is the fix, and the split is not a compromise: it is what each
stage actually needs. PLAN reasons about audio and gets everything. EMIT is a
transcriber working from a decision already made in prose, so it needs the
schema it is serialising into and nothing else. Dropping the DSP source and the
glossary from EMIT removes 64% of its prefix:

    block               PLAN      EMIT
    role                 607       ~250
    graph schema       1,276     1,276
    per-node params    9,105     9,105
    glossary           1,851         -      descriptors were resolved in stage 1
    DSP source        17,840         -      a transcriber does not reason
    TOTAL             30,680    ~10,600

Those figures are from the forty-term glossary. At sixty terms the glossary block
measures 3,616 tokens by the same estimator and the PLAN prefix 33,440 — inside
the 26k-36k band the test asserts, with about 2,500 tokens of headroom left. The
other rows have not been re-measured, so the table is kept as it was rather than
made to add up to a number nobody counted.

Watch `repair_rate` after any change here. If EMIT starts emitting values the
validators have to fix, the missing context is the first suspect.

WHY THE DSP SOURCE CODE IS IN THE PROMPT. It is the single highest-leverage item
in it. Putting the effect implementations in front of the model is what moved
LLM2Fx from 0.53 to 0.22 on EQ matching — the model reasons about what a
parameter DOES rather than about what its name suggests. It is also most of the
prefix's size, which is precisely why the prefix has to cache.

FILE METADATA NEVER REACHES HERE. ID3 tags and Vorbis comments are
attacker-controlled text on a path that terminates in a prompt. The measurement
report describes samples and only samples, and this module has no other source
of text about the audio.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from backend.plan.glossary import as_prompt_table
from backend.schemas.graph import json_schema, param_schemas

_ENGINE_NODES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "engine", "include", "anv", "nodes",
)

# Which node implementations to show, and in this order. Not all eighteen: the
# utility nodes are self-explanatory from their names and their source would cost
# prefix tokens for nothing. These six are the ones whose parameters are
# genuinely hard to reason about from the name alone.
SOURCE_NODES = (
    "ParametricEQ.h",
    "Compressor.h",
    "Drive.h",
    "Delay.h",
    "PlateReverb.h",
    "StereoWidth.h",
)

ROLE = """\
You are the planning stage of an offline audio processing engine. You decide \
what processing to apply to a single audio track, and you never hear the audio.

Instead you are given MEASUREMENTS of it. Reason from those numbers. Do not \
speculate about what the recording sounds like beyond what the measurements \
support, and do not assume anything about the instrument, arrangement or genre \
that the classification does not state.

Three things matter more than they might seem to.

CHAIN ORDER IS PART OF THE ANSWER, not an implementation detail. A compressor \
before an EQ boost and after it are different processors. State the order \
deliberately and say why.

EVERY MOVE MUST BE JUSTIFIED BY A MEASUREMENT. "Vocals usually want a high-pass \
at 100 Hz" is not a justification; "the 60-100 Hz bands sit 4 dB above this \
source's own average and carry low crest, which is rumble rather than \
fundamentals" is. If a measurement does not support a move, do not make it.

RESTRAINT IS THE HARDER SKILL. A chain of four well-chosen moves beats a chain \
of nine defensible ones. The total EQ boost across the whole graph is budgeted \
at 12 dB and exceeding it produces something that sounds like an EQ rather than \
like a record.

EVERY GRAPH ENDS WITH A DELIVERY CEILING. Put a `true_peak_limiter` in the \
`master` section of every chain you produce, at -1.0 dBTP unless the request \
gives you a reason to choose otherwise, and put `dither` after it if the \
delivery format is fixed point. Inter-sample peaks clip on conversion whatever \
the rest of the chain intended, so this is not a stylistic choice and not \
something to leave out when the source already has headroom.

Confidence fields: disregard any estimate below 0.4. Those measurements are \
inferences and they are least reliable on exactly the material where they \
matter most.

Never name an artist, band, producer or commercial recording in your output.\
"""

STAGE1_TASK = """\
Work through these four steps in order, in prose. Do not emit JSON.

1. INPUT ANALYSIS. What do the measurements say about this source? Name the \
specific numbers you are relying on. Note anything the warnings array says you \
must not do.

2. MODULE SELECTION. Which effects address the request, and which do you \
deliberately NOT use? Justify each omission as well as each inclusion — a \
reader should be able to tell you considered and rejected the obvious moves.

3. ORDER DETERMINATION. In what sequence, and why? Where two orders would both \
work, say so and pick one.

4. PARAMETER PLANNING. Specific values, each tied to a measurement. Give your \
running total of EQ boost as you go.

Finish with one paragraph a paying engineer would accept as an explanation of \
what you did and why.\
"""

EMIT_ROLE = """\
You are the serialisation stage of an offline audio processing engine. An \
engineering decision about what processing to apply has already been made, in \
prose, by a stage that reasoned about the measurements. Your only job is to \
transcribe that decision into the schema.

You are not the engineer. Do not re-derive, second-guess, improve or complete \
the decision — if it looks wrong to you, transcribe it anyway and let the \
validators reject it, because a validator's objection is legible to the user \
and yours would not be.

Ranges in the schema are enforced after you reply. A value outside one is \
clamped or the graph is rejected, so copy the decided value rather than \
rounding it toward what you expect.

Never name an artist, band, producer or commercial recording in your output.\
"""

STAGE2_TASK = """\
You are a transcriber. The engineering decision has already been made and is \
written above. Serialise it into the schema exactly as described.

Do not reconsider anything. Do not add a node that was not decided on, do not \
drop one that was, and do not adjust a value you think could be better. If the \
reasoning above is ambiguous about a value, choose the one it most nearly \
states rather than the one you would prefer.

Every node needs a stable lowercase id and a one-sentence rationale referring to \
the measurement that motivated it.\
"""


@dataclass
class Prompt:
    """A prompt split at the cache boundary.

    `system` is the cacheable prefix and must be identical across requests.
    `messages` carries everything per-request.
    """

    system: list[dict[str, Any]]
    messages: list[dict[str, Any]]

    def prefix_text(self) -> str:
        return "\n".join(block.get("text", "") for block in self.system)

    def estimated_prefix_tokens(self) -> int:
        """Rough size of the cached prefix.

        2.4 characters per token, not the usual 4. That constant was measured
        against the live API rather than assumed: the canary reported 30,938
        cached tokens for a prefix the 4-char rule put at 18,409, an
        underestimate of 68%.

        The reason is what the prefix contains. Most of it is C++ source and
        JSON schema — punctuation, identifiers and indentation — which tokenises
        far denser than the prose the 4-char heuristic is derived from.

        Still an estimate, and still only used as a sanity check that the prefix
        is worth caching at all. The number that matters comes from the API's
        own usage counters.
        """
        return int(len(self.prefix_text()) / 2.4)


def _read_source(name: str) -> str:
    path = os.path.join(_ENGINE_NODES, name)
    with open(path) as handle:
        return handle.read()


def _dsp_reference() -> str:
    """The effect implementations, as the model's reference for what parameters do.

    Shipped as source rather than as a written summary because a summary is a
    second description that can drift from the code, and because the header
    comments already explain the reasoning — they were written to be read.
    """
    parts = [
        "The implementations of the effects you are configuring follow. Read a "
        "parameter's use in the code rather than inferring it from its name.",
    ]
    for name in SOURCE_NODES:
        parts.append(f"\n===== {name} =====\n{_read_source(name)}")
    return "\n".join(parts)


def _schema_blocks() -> list[dict[str, Any]]:
    """The schema, which both stages need: one to aim at, one to hit."""
    return [
        {"type": "text",
         "text": "The graph schema you will eventually emit:\n\n"
                 + json.dumps(json_schema(), indent=2, sort_keys=True)},
        {"type": "text",
         "text": "Per-node parameter schemas. Ranges are stated in each "
                 "description and ARE enforced after you reply, so a value "
                 "outside one will be clamped or rejected:\n\n"
                 + json.dumps(param_schemas(), indent=2, sort_keys=True)},
    ]


def _cached(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put the breakpoint on the last block, caching everything above it."""
    blocks[-1] = dict(blocks[-1], cache_control={"type": "ephemeral"})
    return blocks


def system_blocks() -> list[dict[str, Any]]:
    """PLAN's cacheable prefix: everything.

    TAKES NO ARGUMENTS, AND THAT IS THE POINT. Anything per-request that reached
    this function would move the cache boundary before it and turn every
    subsequent token into a miss — silently, since the request still succeeds and
    only the bill changes. A test asserts two independent calls are byte-equal.
    """
    return _cached([
        {"type": "text", "text": ROLE},
        *_schema_blocks(),
        {"type": "text",
         "text": "Descriptor glossary. The `check first` column is a condition "
                 "on the MEASUREMENTS that must hold before the conventional "
                 "move is justified; if it does not hold, say so instead of "
                 "making the move:\n\n" + as_prompt_table()},
        {"type": "text", "text": _dsp_reference()},
    ])


def emit_system_blocks() -> list[dict[str, Any]]:
    """EMIT's cacheable prefix: the schema, and nothing else.

    DELIBERATELY NOT THE SAME PREFIX AS PLAN'S. They run on different models, so
    a shared prefix is written to two caches and read from neither — the cost
    paid twice for a saving collectable once.

    What is dropped is dropped because a transcriber cannot use it. The DSP
    source is in PLAN's prefix so the model can reason about what a parameter
    does rather than what its name suggests; this stage is forbidden from
    reasoning and is working from a decision already written out in prose. The
    glossary resolves descriptors like "boxy" into bands, which stage 1 already
    did. Keeping either here would cost 19,700 tokens per call to supply
    context for a judgement that has already been made.

    Also takes no arguments, for the same reason.
    """
    return _cached([{"type": "text", "text": EMIT_ROLE}, *_schema_blocks()])


def _report_for_prompt(report: dict[str, Any]) -> str:
    """The measurement report, trimmed of what the model cannot use.

    Band lists are kept — they are how a bell gets localised, and the sensitivity
    analysis showed the summary statistics cannot substitute. The ERB duplicate
    is dropped: it answers the same question as the third-octave list for a
    reader that thinks in named frequencies, at twice the token cost.
    """
    trimmed = json.loads(json.dumps(report))
    spectrum = trimmed.get("spectrum")
    if isinstance(spectrum, dict):
        spectrum.pop("erb_db", None)
    return json.dumps(trimmed, indent=2, sort_keys=True)


def plan_prompt(
    *,
    report: dict[str, Any],
    request: str,
    classification: dict[str, Any] | None = None,
    presets: list[dict[str, Any]] | None = None,
) -> Prompt:
    """Stage 1: unconstrained reasoning.

    Unconstrained because constrained decoding from the first token measurably
    degrades reasoning, and removing chain-of-thought costs 13 points of module
    accuracy. The output is prose and is never parsed programmatically — only
    handed to stage 2.
    """
    parts: list[str] = []

    if presets:
        parts.append(
            "Relevant preset dossiers. These are EVIDENCE AND PRIORS, not "
            "recipes: compose them against the measurements rather than "
            "applying them. A preset that hard-codes a 3 kHz boost is wrong for "
            "a source that already has a 3 kHz resonance.\n\n"
            + json.dumps(presets, indent=2, sort_keys=True)
        )

    if classification:
        parts.append(
            "Classification. Every entry carries a probability; treat anything "
            "below 0.4 as unknown and plan for both cases or neither.\n\n"
            + json.dumps(classification, indent=2, sort_keys=True)
        )

    parts.append("Measurement report:\n\n" + _report_for_prompt(report))

    # The user's own words come LAST, immediately before the task. Instructions
    # closest to the end of the prompt carry the most weight, and the request is
    # the thing that must not be forgotten.
    parts.append(f"The user's request, verbatim:\n\n{request.strip()}")
    parts.append(STAGE1_TASK)

    return Prompt(
        system=system_blocks(),
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
    )


def emit_prompt(
    *,
    report: dict[str, Any],
    request: str,
    reasoning: str,
) -> Prompt:
    """Stage 2: constrained serialisation of a decision already made.

    Deliberately a SEPARATE call rather than a continuation. The system prompt
    reframes the model as a transcriber, which it cannot be while the same turn
    still holds the reasoning task — and the schema constraint then applies to a
    turn that has no reasoning left to degrade.
    """
    parts = [
        "Measurement report, for reference while transcribing:\n\n"
        + _report_for_prompt(report),
        f"The user's request, verbatim:\n\n{request.strip()}",
        "The engineering decision to transcribe:\n\n" + reasoning.strip(),
        STAGE2_TASK,
    ]

    return Prompt(
        system=emit_system_blocks(),
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
    )


def repair_prompt(
    *,
    previous: Prompt,
    graph_json: str,
    findings: list[str],
) -> Prompt:
    """A bounded retry, told exactly what was wrong.

    The findings are quoted verbatim from the validators rather than paraphrased.
    A validator's message already names the measurement and the rule; restating
    it in the prompt's own words is an opportunity to soften or mis-state it.
    """
    parts = list(previous.messages[0]["content"].split("\n\n"))
    parts.append(
        "That graph was rejected. The problems, verbatim from the validators:\n\n"
        + "\n".join(f"- {finding}" for finding in findings)
    )
    parts.append(
        "Your previous output:\n\n" + graph_json
    )
    parts.append(
        "Emit a corrected graph. Change only what the findings require — every "
        "other decision was accepted and rewriting it discards work the user "
        "has not objected to."
    )

    return Prompt(
        system=previous.system,
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
    )
