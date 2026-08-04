# Backend test runner.
#
# Plain make rather than pytest: the suites print a progress line per section
# and a check count, matching the engine's C++ harness, and the whole point of
# that format is that the two halves of the project report the same way.
PY ?= ../.venv/bin/python

.PHONY: test loudness measure bridge sensitivity recovery schema planner rules refine dynamics verify api listen sweep recover recover-synthetic manifest knowledge

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
	@$(PY) -m backend.eval.run_sensitivity $(STEMS)

# Chain recovery: which node types and what order can be read off the deltas.
# The hermetic half asserts the scoring; `recover` prints the real floor that
# any future planner has to beat.
recovery:
	@$(PY) tests/test_recovery.py

recover:
	@$(PY) -m backend.eval.run_recovery $(STEMS)

# The same run without the stems, which are gitignored. Not a substitute for
# real material — pink noise already has energy everywhere, which masks exactly
# the effect drive detection depends on — but it exercises the whole harness on
# a machine that has no audio.
recover-synthetic:
	@$(PY) -m backend.eval.run_recovery --synthetic --quiet

# Rewrite the committed corpus manifest. Only after a deliberate change to the
# generator: every number in the report moves when this does, which is the whole
# reason the manifest is committed.
manifest:
	@$(PY) -m backend.eval.generate

# The planner's output contract and its semantic validators. Reads the engine
# headers to assert the schema has not drifted from what the engine accepts.
schema:
	@$(PY) tests/test_schema.py

# The scaffolding around the two model calls: quantifier ladder, glossary,
# prompt cache boundary, and the two-stage driver with its repair loop. Runs
# entirely offline against a scripted client.
planner:
	@$(PY) tests/test_planner.py

# Match-EQ. Recovers a known filter applied by the real engine, and checks the
# analytic response the optimiser fits against agrees with what the engine renders.
# The deterministic planner. Mostly tests refusals: any set of rules can find
# something to do to every file, and the hard part is leaving a bass alone.
rules:
	@$(PY) tests/test_rules.py

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
