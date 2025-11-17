install:
	pip install uv &&\
	uv sync

test:
	uv run python -m pytest tests/ -vv --cov=logic --cov=api --cov=cli 

format:	
	uv run black logic/*.py cli/*.py api/*.py #*.py

lint:
	uv run pylint --disable=R,C --ignore-patterns=test_.*\.py logic/*.py cli/*.py api/*.py 

refactor: format lint

all: install lint test format