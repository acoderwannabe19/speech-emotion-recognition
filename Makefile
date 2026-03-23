.PHONY: help install lint fix format test clean mlflow run

VENV := .venv/bin/

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	$(VENV)pip install -r requirements.txt

lint: ## Run ruff linter (check only)
	$(VENV)ruff check src/ tests/

fix: ## Run ruff linter with auto-fix
	$(VENV)ruff check --fix .

format: ## Format code with ruff
	$(VENV)ruff format .

test: ## Run tests with pytest
	$(VENV)pytest

clean: ## Remove caches and build artefacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

mlflow: ## Launch MLflow UI
	$(VENV)mlflow ui --backend-store-uri models/mlruns

run: ## Launch Streamlit app
	$(VENV)streamlit run frontend/app.py
