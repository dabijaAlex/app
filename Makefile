# Makefile - serial Longstaff-Schwartz American option pricer.
#   make          builds bin/lsm
#   make test     builds and runs bin/lsm_tests (non-zero exit on failure)
#   make debug    -O0 -g build (+ sanitizers on Linux; not available on MinGW)
#   make clean
# Works with GNU make on Linux and mingw32-make (MSYS2 UCRT64) on Windows.

CXX      = g++
CXXFLAGS = -std=c++23 -O3 -march=native -Wall -Wextra -Wpedantic -Iinclude
# Debug build: -O0 -g -fsanitize=address,undefined
# DO NOT use -ffast-math: it breaks reproducibility, which later parallel versions rely on.
LDLIBS   =

ifeq ($(OS),Windows_NT)
EXE      = .exe
# MinGW's libstdc++ keeps the Windows-console part of std::print in libstdc++exp.
LDLIBS  += -lstdc++exp
# Link the runtime statically so bin/lsm.exe runs without MSYS2 DLLs on PATH.
LDLIBS  += -static
SANITIZE =
else
EXE      =
SANITIZE = -fsanitize=address,undefined
endif

LIB_SRC  = src/config.cpp src/paths.cpp src/regression.cpp src/lsm.cpp src/reference.cpp src/report.cpp
LIB_OBJ  = $(patsubst src/%.cpp,build/%.o,$(LIB_SRC))
HEADERS  = $(wildcard include/lsm/*.hpp)

BIN      = bin/lsm$(EXE)
TEST_BIN = bin/lsm_tests$(EXE)

# Shell commands. mingw32-make falls back to cmd.exe (and leaves SHELL as the
# bare string "sh.exe") when no POSIX sh is on PATH, e.g. from PowerShell.
ifeq ($(SHELL),sh.exe)
MKDIR    = if not exist $(1) mkdir $(1)
RMDIR    = if exist build rmdir /s /q build & if exist bin rmdir /s /q bin
RUN      = $(subst /,\,$(1))
else
MKDIR    = mkdir -p $(1)
RMDIR    = rm -rf build bin
RUN      = ./$(1)
endif

.PHONY: all test clean debug run-bench

all: $(BIN)

build/%.o: src/%.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/test_main.o: tests/test_main.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN): build/main.o $(LIB_OBJ) | bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(TEST_BIN): build/test_main.o $(LIB_OBJ) | bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

test: $(TEST_BIN)
	$(call RUN,$(TEST_BIN))

debug: CXXFLAGS = -std=c++23 -O0 -g -Wall -Wextra -Wpedantic -Iinclude $(SANITIZE)
debug: clean all

run-bench: $(BIN)
	$(call RUN,$(BIN)) --preset bench --input data/options_benchmark.csv

build bin:
	$(call MKDIR,$@)

clean:
	$(RMDIR)
