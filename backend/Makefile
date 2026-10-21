# Backend test runner.
#
# Plain make rather than pytest: the suites print a progress line per section
# and a check count, matching the engine's C++ harness, and the whole point of
# that format is that the two halves of the project report the same way.
PY ?= ../.venv/bin/python

# Targets that `cd ..` need the interpreter path from the ROOT, not from here.
# Four eval targets used `-m backend.eval.X` while make runs in backend/, so the
# package was not importable and none of them had ever run — README documented
# the failing invocation. They cd to the root and use this instead.
ROOT_PY ?= .venv/bin/python

.PHONY: test loudness measure bridge sensitivity recovery schema planner rules refine dynamics verify api listen sweep recover recover-synthetic manifest knowledge author

test: loudness measure bridge sensitivity recovery schema planner rules refine dynamics verify api listen knowledge

loudness:
	@$(PY) tests/test_loudness.py

measure:
	@$(PY) tests/test_measure.py

# Cross-validation against the C++ engine. Needs engine/build/anv_render, so
# it is the one suite with a build dependency outside this directory.
bridge:
	@$(PY) tests/test_engine_bridge.py

# Can the measurement report see what the engine does? Runs on synthetic
# sources so it is hermetic; `make sweep` runs the same analysis over real
# stems and prints the table.
sensitivity:
	@$(PY) tests/test_sensitivity.py

sweep:
	@cd .. && $(ROOT_PY) -m backend.eval.run_sensitivity $(STEMS)

# Chain recovery: which node types and what order can be read off the deltas.
# The hermetic half asserts the scoring; `recover` prints the real floor that
# any future planner has to beat.
recovery:
	@$(PY) tests/test_recovery.py

recover:
	@cd .. && $(ROOT_PY) -m backend.eval.run_recovery $(STEMS)

# The same run without the stems, which are gitignored. Not a substitute for
# real material — pink noise already has energy everywhere, which masks exactly
# the effect drive detection depends on — but it exercises the whole harness on
# a machine that has no audio.
recover-synthetic:
	@cd .. && $(ROOT_PY) -m backend.eval.run_recovery --synthetic --quiet

# Rewrite the committed corpus manifest. Only after a deliberate change to the
# generator: every number in the report moves when this does, which is the whole
# reason the manifest is committed.
manifest:
	@cd .. && $(ROOT_PY) -m backend.eval.generate

# The planner's output contract and its semantic validators. Reads the engine
# headers to assert the schema has not drifted from what the engine accepts.
schema:
	@$(PY) tests/test_schema.py

# The scaffolding around the two model calls: quantifier ladder, glossary,
# prompt cache boundary, and the two-stage driver with its repair loop. Runs
# entirely offline against a scripted client.
planner:
	@$(PY) tests/test_planner.py

# The deterministic planner. Mostly tests refusals: any set of rules can find
# something to do to every file, and the hard part is leaving a bass alone.
rules:
	@$(PY) tests/test_rules.py

# Match-EQ. Recovers a known filter applied by the real engine, and checks the
# analytic response the optimiser fits against agrees with what the engine renders.
refine:
	@$(PY) tests/test_refine.py

# The post-render safety gate. Tests both directions of every check: the thing
# it must catch, and the very similar thing it must not.
dynamics:
	@$(PY) tests/test_match_dynamics.py

verify:
	@$(PY) tests/test_verify.py

# The preset knowledge base. The catalogue is empty on purpose; what this suite
# checks is that an unsourced factual claim cannot be REPRESENTED, so the empty
# state cannot quietly become a half-cited one.
knowledge:
	@$(PY) tests/test_knowledge.py

# The preset authoring tool, which is the mitigation for the plan's risk #7:
# authoring is the schedule bottleneck and is not an engineering task, so
# everything an engineer can do to shorten it is tooling.
#
#   make author                                       the help, where to start
#   make author AUTHOR_ARGS="check"                   validate the catalogue
#   make author AUTHOR_ARGS="new -o d.json"           interactive scaffold
#   make author AUTHOR_ARGS="audition d.json s.wav -o out"    hear the priors
#   make author AUTHOR_ARGS="diff a.json b.json s.wav"        what separates them
#
# NOT part of `test`: `audition` and `diff` need a source file and the engine
# binary, and `check` over an empty catalogue asserts nothing. The properties
# are tested in tests/test_knowledge.py, which synthesises its own audio.
# Invoked as a FILE rather than with `-m`, like the test targets. Make runs
# with this directory as the working directory, so `python -m
# backend.knowledge.author` cannot resolve the `backend` package; the `-m`
# targets above work only because they `cd ..` first.
AUTHOR_ARGS ?= --help

author:
	@$(PY) knowledge/author.py $(AUTHOR_ARGS)

# The local API, driven through TestClient. Synthesises its own audio so it does
# not need the gitignored stems, and only ever plans with `rules`, so no suite
# in this Makefile can make a model call.
api:
	@$(PY) tests/test_api.py

# The blinded listening harness. THE ONE SUITE THAT GATES EVERY QUALITY CLAIM:
# no automated metric may gate CI until it correlates with human MOS at
# rho > 0.6, and nothing can be correlated against human judgement until there
# is a way to collect it. Renders its screening stimulus with the real engine
# and starts a local HTTP server on an ephemeral port, so it is grouped with
# the slower suites.
listen:
	@$(PY) tests/test_listen.py
