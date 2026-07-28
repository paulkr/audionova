# Audionova render engine — Phase 0 build.
#
# Deliberately a plain Makefile for now: the engine is header-only plus a test
# binary, and this keeps the loop fast with zero toolchain prerequisites.
# Migrates to CMake + scikit-build-core when the pybind11 wheel lands, since
# cibuildwheel needs it.
#
# Flag notes, all of which affect correctness rather than speed:
#   -ffp-contract=off   FMA contraction changes rounding, and would make the
#                       same source produce different audio on different
#                       compilers. Reproducibility outranks the free multiply.
#   NO -ffast-math      It permits reassociation and assumes no NaN/Inf, both
#                       of which break the guarantees the contract tests check.
#   -fno-math-errno     Safe, and lets the compiler inline libm calls.

CXX      ?= clang++
CXXFLAGS := -std=c++20 -O2 -g \
            -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion \
            -ffp-contract=off -fno-math-errno \
            -Iinclude

BUILD    := build
TESTBIN  := $(BUILD)/contract_tests
SANBIN   := $(BUILD)/contract_tests_san

.PHONY: all test san check clean

all: $(TESTBIN)

# Header dependencies come from the compiler (-MMD -MP), not from a `find`
# glob. A glob evaluated at parse time can miss an edit and hand `make test` a
# stale binary — which for this project means a broken engine passing CI. The
# depfile makes that impossible.
$(TESTBIN): tests/contract_tests.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -MMD -MP -MF $(BUILD)/contract_tests.d \
	    tests/contract_tests.cpp -o $@

-include $(BUILD)/contract_tests.d

test: $(TESTBIN)
	@./$(TESTBIN)

# Address + UB sanitizers. Not optional discipline: mutation testing found a
# partition-indexing bug that read out of bounds, produced benign-looking
# garbage, and passed every functional assertion. ASan caught it instantly.
# Memory errors do not reliably show up as wrong numbers.
#
# Built at -O1 without the reproducibility flags, because this target is
# hunting memory errors rather than validating audio output.
$(SANBIN): tests/contract_tests.cpp
	@mkdir -p $(BUILD)
	$(CXX) -std=c++20 -O1 -g -fsanitize=address,undefined \
	    -fno-omit-frame-pointer -Iinclude \
	    -MMD -MP -MF $(BUILD)/contract_tests_san.d \
	    tests/contract_tests.cpp -o $@

-include $(BUILD)/contract_tests_san.d

san: $(SANBIN)
	@./$(SANBIN)

# What CI should run.
check: test san

clean:
	@rm -rf $(BUILD)
