# Backend test runner.
#
# Plain make rather than pytest: the suites print a progress line per section
# and a check count, matching the engine's C++ harness, and the whole point of
# that format is that the two halves of the project report the same way.
PY ?= ../.venv/bin/python

.PHONY: test loudness measure bridge sensitivity recovery sweep recover

test: loudness measure bridge sensitivity recovery

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
