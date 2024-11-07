PY ?= python

.PHONY: build test demo live-demo bench clean
build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$$(command -v $(PY))
	cmake --build build -j
test: build
	$(PY) -m pytest -q
demo:
	$(PY) -m circuitscope all
live-demo:
	$(PY) -m circuitscope demo
bench:
	$(PY) -m circuitscope bench
clean:
	rm -rf build artifacts circuitscope/_kernels*.so .pytest_cache
