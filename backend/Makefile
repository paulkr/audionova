# Backend test runner.
#
# Plain make rather than pytest: the suites print a progress line per section
# and a check count, matching the engine's C++ harness, and the whole point of
# that format is that the two halves of the project report the same way.
PY ?= ../.venv/bin/python

.PHONY: test loudness measure

test: loudness measure bridge

loudness:
	@$(PY) tests/test_loudness.py

measure:
	@$(PY) tests/test_measure.py

# Cross-validation against the C++ engine. Needs engine/build/anv_render, so
# it is the one suite with a build dependency outside this directory.
bridge:
	@$(PY) tests/test_engine_bridge.py
