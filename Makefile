# Development and pipeline entry points.
# Run `make help` for the list.

PYTHON  ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
CONFIG  ?= configs/default.yaml

.DEFAULT_GOAL := help
.PHONY: help setup install lint format test test-fast coverage results \
        generate validate eda train clv explain experiments all \
        api dashboard docker-build docker-run clean clean-outputs

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv and install the package with dev extras
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e .
	@echo "Activate with: source $(VENV)/bin/activate"

install: ## Install the package into the current environment
	pip install -e .

lint: ## Check formatting and lint rules
	$(BIN)/ruff check src tests api dashboard
	$(BIN)/ruff format --check src tests api dashboard

format: ## Apply formatting and auto-fixable lint rules
	$(BIN)/ruff format src tests api dashboard
	$(BIN)/ruff check --fix src tests api dashboard

test: ## Run the whole test suite
	$(BIN)/pytest

test-fast: ## Run only the tests that do not train models
	$(BIN)/pytest -m "not slow"

coverage: ## Run tests with a coverage report
	$(BIN)/pytest --cov=returns_clv --cov-report=term-missing --cov-report=html

# --- pipeline stages -------------------------------------------------------

generate: ## Regenerate the synthetic dataset (writes a new file, never overwrites)
	$(BIN)/returns-clv generate --config $(CONFIG) --out data/synthetic_ecommerce_regenerated.csv

validate: ## Run data quality and business-logic checks
	$(BIN)/returns-clv validate --config $(CONFIG)

eda: ## Exploratory analysis figures and tables
	$(BIN)/returns-clv eda --config $(CONFIG)

train: ## Train, calibrate, choose a threshold and evaluate
	$(BIN)/returns-clv train --config $(CONFIG)

clv: ## Score customers and value the portfolio
	$(BIN)/returns-clv clv --config $(CONFIG)

explain: ## SHAP feature attributions
	$(BIN)/returns-clv explain --config $(CONFIG)

experiments: ## Quantify the methodology fixes (leakage, threshold optimism, ablation)
	$(BIN)/returns-clv experiments --config $(CONFIG)

all: ## Run the full pipeline end to end, then refresh the documented results
	$(BIN)/returns-clv all --config $(CONFIG)
	$(BIN)/returns-clv experiments --config $(CONFIG)
	$(MAKE) results

results: ## Regenerate the results tables in README.md and docs/ from outputs/reports
	$(BIN)/python scripts/update_results.py

# --- serving ---------------------------------------------------------------

api: ## Serve the scoring API on http://localhost:8000 (docs at /docs)
	$(BIN)/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Serve the Streamlit dashboard on http://localhost:8501
	$(BIN)/streamlit run dashboard/app.py

docker-build: ## Build the API container image
	docker build -t returns-clv:latest .

docker-run: ## Run the API container against the local models directory
	docker run --rm -p 8000:8000 -v "$(PWD)/models:/app/models:ro" returns-clv:latest

# --- housekeeping ----------------------------------------------------------

clean-outputs: ## Delete generated figures, reports and models
	rm -rf outputs/figures/*.png outputs/reports/* models/*.joblib

clean: clean-outputs ## Also delete caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov *.egg-info src/*.egg-info
