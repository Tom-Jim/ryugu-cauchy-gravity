.DEFAULT_GOAL := help

.PHONY: help install build build-release serve check test format cpp clean

help:
	@printf '%s\n' \
	  'install        Install Bun dependencies' \
	  'build          Build WASM and the Vue bundle for development' \
	  'build-release  Build the optimized static package' \
	  'serve          Serve static files only' \
	  'check          Typecheck, format-check and lint' \
	  'test           Run the Rust test suite' \
	  'format         Format Rust sources' \
	  'cpp            Build the optional ESA C++ reference bridge' \
	  'clean          Remove generated build output'

install:
	bun install

build:
	bun run build:dev

build-release:
	bun run build:release

serve:
	bun run serve

check:
	bun run typecheck
	cargo fmt --all --check
	cargo clippy --all-targets -- -D warnings

test:
	cargo test

format:
	cargo fmt --all

cpp:
	cmake -S src/c++ -B build/c++ -DCMAKE_BUILD_TYPE=Release
	cmake --build build/c++ --parallel

clean:
	rm -rf build dist node_modules pkg target
