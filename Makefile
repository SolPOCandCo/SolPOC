SOURCES = $(wildcard solpoc/*) $(wildcard solpoc/*/*)
METADATA = pyptoject.toml MANIFEST.in README.md LICENSE

docs: docs-html

docs-html:
	pdoc --html --force -o docs/html solpoc
	pdoc --pdf solpoc > docs/solpoc_doc.md

requirements:
	python -m pip install -r requirements.txt

test: requirements
	python -m pytest

majorversion:
	bump2version major 

minorversion:
	bump2version minor 

patchversion:
	bump2version patch 

build: $(SOURCES) $(METADATA) 
	python -m build

install:
	pip install . --force-reinstall --no-deps

test-upload: build
	twine upload --non-interactive --repository testpypi --config-file .pypirc dist/*

upload: build
	twine upload --non-interactive --repository pypi --config-file .pypirc dist/*

clean:
	if (Test-Path dist) {rm -r -force dist}
	if (Test-Path build) {rm -r -force build}

$(SOURCES):

$(METADATA):