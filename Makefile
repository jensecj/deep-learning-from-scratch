.PHONY: default
default: check test

check:
	mypy dlfs

test:
	pytest

install:
	pip install -e .

clean:
	rm -r dist build dlfs.egg-info __pycache__ .mypy_cache .pytest_cache .cache .eggs

deps:
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt
