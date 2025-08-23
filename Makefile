PYTHON = python
PIP = pip
VENV = .venv
ACTIVATE = source $(VENV)/bin/activate

setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && $(PIP) install -U pip
	$(ACTIVATE) && $(PIP) install -e ".[dev]"
	$(ACTIVATE) && pre-commit install

ingest:
	$(ACTIVATE) && $(PYTHON) -m src.data.ingest

features:
	$(ACTIVATE) && $(PYTHON) -m src.features.build_features

train:
	$(ACTIVATE) && $(PYTHON) -m src.models.train

signals:
	$(ACTIVATE) && $(PYTHON) -m src.strategy.generate_signals

backtest:
	$(ACTIVATE) && $(PYTHON) -m src.backtest.simulator

run_all: ingest features train signals backtest

test:
	$(ACTIVATE) && pytest -q

lint:
	$(ACTIVATE) && ruff check .
	$(ACTIVATE) && black --check .

format:
	$(ACTIVATE) && ruff check . --fix
	$(ACTIVATE) && black .

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	find . -type f -name "*.pyc" -delete
