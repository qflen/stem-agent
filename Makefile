.PHONY: run eval test lint journal clean install

install:
	uv pip install -e ".[dev]"

run:
	stem-agent differentiate --domain code_quality

eval:
	stem-agent evaluate

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

journal:
	stem-agent journal --last

clean:
	rm -rf .ruff_cache __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
