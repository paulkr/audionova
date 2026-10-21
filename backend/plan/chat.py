"""Free text into patches. The model names the target; Python does the maths.

WHAT THIS IS FOR. After a chain exists and the user has heard it, the next thing
they want is not another plan — it is "decrease the delay by 15%" or "a bit more
bass". Handing that to the planner would regenerate the whole graph and churn
every decision they implicitly accepted by not complaining about it. To a user
that reads as a bug: they asked for one change and six things moved.

So this is a PARSER, not a planner. Its entire output is a list of patches drawn
from the closed set in `ladder.py`, applied by arithmetic that lives in Python.

    "decrease the delay by 15%"   -> adjust time_ms, decrease, percent=15
    "a bit more bass"             -> adjust the low shelf's gain, quantifier=a_bit
    "make the reverb longer"      -> adjust decay_s, increase, quantifier=slightly

THE MODEL NEVER COMPUTES A VALUE, and this is the same invariant `ladder.py`
rests on rather than a new one. It extracts "15" and "percent" and "delay" from
the sentence. Every multiplication happens in `adjust_by_amount`. A model asked
to apply "slightly" gives 1.2 dB one turn and 4 dB the next, and the user
concludes the tool is unpredictable rather than that they were unclear.

WHY IT IS A SEPARATE, CHEAP CALL. The planner's stage 1 takes three minutes
because it is reasoning about audio from measurements. This is reading one
sentence against a graph that already exists, so it is a small prompt, a small
output, and a fast model. A refinement that took three minutes would not be a
refinement.

WHAT IT REFUSES. A sentence naming something the chain does not have — "less
reverb" on a chain with no reverb — is refused with what IS there, rather than
adding a reverb the user did not ask for. Adding a node is a planning decision;
this stage only moves what exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from backend.plan import ladder
from backend.plan.ladder import Patch
from backend.schemas.graph import EffectGraph

# The parser's whole vocabulary. Deliberately the ladder's own ops rather than a
# parallel set — a second vocabulary would drift from the first, and the first
# is the one with the arithmetic behind it.
CHAT_OPS = ("adjust_param", "bypass_node", "enable_node")


@dataclass
class ChatResult:
    patches: list[Patch] = field(default_factory=list)
    reply: str = ""
    refused: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.patches) and not self.refused


def graph_summary(graph: EffectGraph) -> str:
    """What the model is shown: the chain, with ids and current values.

    Ids matter more than anything else here. A patch names a node by id, so a
    summary without them would force the model to invent one, and an invented id
    is a patch that silently applies to nothing.
    """
    # THE ID AND THE BAND INDEX MUST NOT LOOK LIKE ONE STRING. The first
    # version printed "tone band 0: lowshelf at 180 Hz", and the model dutifully
    # reported node_id="tone band 0" — which matches no node, so every EQ edit
    # failed with "no node with id 'tone band 0'". The model read exactly what
    # was written; the format was the bug.
    lines = []
    for node in list(graph.nodes) + list(graph.master):
        bands = node.params.get("bands")
        if isinstance(bands, list) and bands:
            lines.append(f'  node_id="{node.id}" ({node.type})'
                         + ("  [bypassed]" if node.bypass else ""))
            for index, band in enumerate(bands):
                lines.append(
                    f"      band_index={index}: {band.get('filter_type')}, "
                    f"center_freq_hz={band.get('center_freq_hz')}, "
                    f"gain_db={band.get('gain_db')}, "
                    f"q_factor={band.get('q_factor')}")
        else:
            values = ", ".join(
                f"{k}={v}" for k, v in node.params.items()
                if isinstance(v, (int, float)))
            lines.append(f'  node_id="{node.id}" ({node.type}): '
                         f"{values or 'no numeric parameters'}, mix={node.mix}"
                         + ("  [bypassed]" if node.bypass else ""))
    return "\n".join(lines) or "  (the chain is empty)"


SYSTEM = """\
You turn one sentence from an audio engineer into edits to a chain that already \
exists. You are a parser, not a planner.

YOU NEVER COMPUTE A NEW VALUE. You name the parameter, the direction, and the \
size the user gave. Python applies it. If the user said "15%" you report \
percent 15; you do not report the resulting number.

THE SIZE COMES FROM THE SENTENCE, in one of three forms:
  percent   the user gave a percentage: "by 15%", "twice as much" is 100
  amount    the user gave a number in the parameter's own unit: "2 dB more"
  word      the user was qualitative: slightly, a_bit, noticeably, a_lot

Use `word` when the sentence is qualitative. Do not convert a vague phrase into \
a number you invented — "a bit more" is `a_bit`, not 10 percent.

ONLY MOVE WHAT EXISTS. Every edit names a `node_id` EXACTLY as it appears in \
the chain below — the quoted string, never the line it was printed on. A band \
is addressed by its node's id plus `band_index`, which are two separate fields.

ONLY MOVE WHAT EXISTS. Every edit names a node id from the chain you are shown. \
If the user asks for something the chain does not have, refuse and say what is \
there instead. Adding a new effect is a planning decision and not yours.

AMBIGUITY RULES, because these words are not as clear as they feel:
  "more delay"      the delay is LOUDER: its mix
  "more repeats"    the feedback
  "longer delay"    the time
  "more bass"       the lowest-frequency boost in the chain, or a low shelf
  "brighter"        a high shelf or the highest band
  "more compression" a LOWER threshold

If a sentence could mean two parameters and you cannot tell, refuse and ask \
which — a wrong guess costs the user a render and their trust in the next one.\
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["edits", "reply"],
    "properties": {
        "edits": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["node_id", "param", "direction", "size_kind"],
                "properties": {
                    "node_id": {"type": "string"},
                    # A CLOSED SET, for the same reason node types are one.
                    # Left open, the model wrote `gain` where the field is
                    # `gain_db` — a name that looks right, does not exist, and
                    # fails at apply time with an error about the schema rather
                    # than about the sentence. Every name here has a declared
                    # perceptual step behind it, so a parameter the ladder
                    # cannot move is not offerable in the first place.
                    "param": {"enum": sorted(ladder.STEPS)},
                    "band_index": {"type": "integer"},
                    "direction": {"enum": ["increase", "decrease"]},
                    "size_kind": {"enum": ["percent", "amount", "word"]},
                    "percent": {"type": "number"},
                    "amount": {"type": "number"},
                    "word": {"enum": ["slightly", "a_bit", "noticeably",
                                      "a_lot"]},
                    "reason": {"type": "string"},
                },
            },
        },
        "reply": {
            "type": "string",
            "description": "One sentence to the user saying what you changed, "
                           "in their words. Never a number you calculated.",
        },
        "refusal": {
            "type": "string",
            "description": "Set instead of edits when the request names "
                           "something the chain does not have, or is ambiguous. "
                           "Say what IS there.",
        },
    },
}


def _unreadable(edit: dict[str, Any]) -> str:
    """Why one edit cannot be applied as written, or "" if it can.

    Checked against SCHEMA's own enums, so the two cannot drift: structured
    output is not guaranteed to honour an enum, and this is where that lands.
    """
    if not isinstance(edit, dict):
        return f"an edit is {type(edit).__name__}, not an object"
    fields = SCHEMA["properties"]["edits"]["items"]["properties"]
    for name in ("direction", "size_kind"):
        if edit.get(name) not in fields[name]["enum"]:
            return f"{name} is {edit.get(name)!r}, not one of {fields[name]['enum']}"
    kind = edit["size_kind"]
    if kind == "word" and edit.get("word") not in fields["word"]["enum"]:
        return f"word is {edit.get('word')!r}, not one of {fields['word']['enum']}"
    if kind in ("percent", "amount"):
        value = edit.get(kind)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"size_kind is {kind!r} but {kind} is {value!r}"
    return ""


def parse(client: Any, *, message: str, graph: EffectGraph) -> ChatResult:
    """One sentence and a chain in; patches out."""
    prompt = (
        f"The chain, with the ids you must use:\n{graph_summary(graph)}\n\n"
        f"The engineer said, verbatim:\n\n{message.strip()}"
    )
    raw = client.complete_structured(
        [{"type": "text", "text": SYSTEM}],
        [{"role": "user", "content": prompt}],
        SCHEMA,
    )

    from backend.plan.pipeline import extract_json
    try:
        parsed = extract_json(raw)
    except Exception as error:
        return ChatResult(refused=f"could not read the reply: {error}")

    if parsed.get("refusal"):
        return ChatResult(refused=str(parsed["refusal"])[:400])

    patches: list[Patch] = []
    for edit in parsed.get("edits", []) or []:
        problem = _unreadable(edit)
        if problem:
            # A REFUSAL, NOT A DEFAULT. This used to read any direction but
            # "decrease" as "increase", a missing size as "slightly" and a
            # missing number as 0 — each a guess at what the model meant,
            # applied and reported as done. The caller is a language model, so
            # a value outside the schema is a malformed reply, and the user is
            # better served by "say that again" than by the opposite change.
            return ChatResult(refused=f"could not read the reply: {problem}")
        kind = edit["size_kind"]
        patch = Patch(
            op="adjust_param",
            node_id=str(edit.get("node_id", "")),
            param=str(edit.get("param", "")),
            direction=edit["direction"],
            band_index=edit.get("band_index"),
            reason=str(edit.get("reason", ""))[:200],
        )
        if kind == "percent":
            patch.percent = float(edit["percent"])
        elif kind == "amount":
            patch.delta = abs(float(edit["amount"]))
        else:
            patch.quantifier = edit["word"]
        patches.append(patch)

    if not patches:
        return ChatResult(refused="nothing in that named something to change")
    return ChatResult(patches=patches, reply=str(parsed.get("reply", ""))[:400])
