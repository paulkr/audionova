# Top-level entry points. The real work lives in engine/Makefile,
# backend/Makefile and ci/run.sh; this exists so `make` from the repository root
# does the obvious thing.

.PHONY: ci quick engine backend clean

# Everything CI runs, in the same script CI runs.
ci:
	@ci/run.sh

# Skips the sanitizer build, which dominates the wall clock at 40s of a 3m run.
quick:
	@ci/run.sh quick

engine:
	@$(MAKE) -C engine check

backend:
	@$(MAKE) -C backend test

clean:
	@$(MAKE) -C engine clean
	@find backend -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
