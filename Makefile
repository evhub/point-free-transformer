.PHONY: test
test: install
	python -m point_free_transformer

.PHONY: install
install: build
	pip install -Ue .[dev]

.PHONY: build
build: clean
	coconut setup.coco --target 3.6 --strict
	coconut "point_free_transformer-source" point_free_transformer --target 3.6 --strict --mypy

.PHONY: clean
clean:
	rm -rf ./dist ./build ./point_free_transformer

.PHONY: setup
setup:
	pip install -U setuptools wheel pip
	pip install -U coconut-develop[watch,mypy]
