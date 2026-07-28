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
RENDERBIN := $(BUILD)/anv_render
IOTESTBIN := $(BUILD)/io_tests

.PHONY: all test iotest san check render clean

all: $(TESTBIN) $(IOTESTBIN) $(RENDERBIN)

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

# WAV, JSON, resampling and recipe validation. A separate binary from the node
# contract suite because none of it is a node, and parameterizing the node
# registry over these checks would make every future node walk past twenty
# assertions that do not apply to it.
$(IOTESTBIN): tests/io_tests.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -MMD -MP -MF $(BUILD)/io_tests.d \
	    tests/io_tests.cpp -o $@

-include $(BUILD)/io_tests.d

iotest: $(IOTESTBIN)
	@./$(IOTESTBIN)

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

# The render CLI. Same flags as the tests, deliberately: the binary that ships
# audio must be built exactly like the binary that validates it, or the tests
# are validating a different program.
$(RENDERBIN): src/anv_render.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -MMD -MP -MF $(BUILD)/anv_render.d \
	    src/anv_render.cpp -o $@

-include $(BUILD)/anv_render.d

render: $(RENDERBIN)

# What CI should run.
check: test iotest san $(RENDERBIN)

clean:
	@rm -rf $(BUILD)
