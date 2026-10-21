# Top-level entry points. The real work lives in engine/Makefile,
# backend/Makefile and ci/run.sh; this exists so `make` from the repository root
# does the obvious thing.

.PHONY: ci quick engine backend demo clean

# Everything CI runs, in the same script CI runs.
ci:
	@ci/run.sh

# Skips the sanitizer build, about 3 minutes of an 11-minute run.
quick:
	@ci/run.sh quick

# Hear it. Renders a file through a hand-written chain and writes a
# loudness-matched copy for a fair A/B.
#   make demo IN=audio/stems/vocals.wav
demo:
	@demo/run.sh $(IN)

engine:
	@$(MAKE) -C engine check

backend:
	@$(MAKE) -C backend test

clean:
	@$(MAKE) -C engine clean
	@find backend -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
