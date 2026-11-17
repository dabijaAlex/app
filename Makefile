# Makefile - serial Longstaff-Schwartz American option pricer (Linux, GCC >= 14).
#   make          builds bin/lsm
#   make test     builds and runs bin/lsm_tests (non-zero exit on failure)
#   make run      prices the example option and draws the plots
#   make debug    -O0 -g build with ASan/UBSan
#   make clean

CXX      = g++
CXXFLAGS = -std=c++23 -O3 -march=native -Wall -Wextra -Wpedantic -Iinclude
# DO NOT use -ffast-math: it breaks reproducibility, which later parallel versions rely on.

LIB_SRC  = src/paths.cpp src/regression.cpp src/lsm.cpp
LIB_OBJ  = $(patsubst src/%.cpp,build/%.o,$(LIB_SRC))
HEADERS  = $(wildcard include/lsm/*.hpp)

BIN      = bin/lsm
TEST_BIN = bin/lsm_tests

.PHONY: all test run clean debug

all: $(BIN)

build/%.o: src/%.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/test_main.o: tests/test_main.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN): build/main.o $(LIB_OBJ) | bin
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TEST_BIN): build/test_main.o $(LIB_OBJ) | bin
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

# Example: first row of the Longstaff-Schwartz (2001) table.
run: $(BIN)
	./$(BIN) put 36 40 0.06 0.2 1
	python3 scripts/plot.py out

debug: CXXFLAGS = -std=c++23 -O0 -g -Wall -Wextra -Wpedantic -Iinclude -fsanitize=address,undefined
debug: clean all

build bin:
	mkdir -p $@

clean:
	rm -rf build bin
