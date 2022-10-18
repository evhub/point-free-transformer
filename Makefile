.PHONY: test
test: highlight
	python -m point_free_transformer

.PHONY: highlight
highlight: install
	pygmentize -f html -S default -a .highlight > pygments.css
	pygmentize -f html -l coconut -o raw_highlight.html "./point_free_transformer-source/__init__.coco"
	python -m point_free_transformer.highlight

.PHONY: install
install: build
	pip install -Ue .

.PHONY: build
build: clean
	coconut setup.coco --target 3.6 --strict
	coconut "point_free_transformer-source" point_free_transformer --target 3.6 --strict --mypy

.PHONY: clean
clean:
	rm -rf ./dist ./build

.PHONY: wipe
wipe:
	rm -rf ./point_free_transformer

.PHONY: setup
setup:
	pip install -U setuptools wheel pip
	pip install -U coconut-develop[watch,mypy]
