.PHONY: help install dev-install clean test lint format type-check pre-commit run-tests coverage docs build publish benchmark

help: ## Show this help message
	@echo "GeoDaedalus Development Commands"
	@echo "================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

dev-install: ## Install development dependencies
	uv sync --all-extras

clean: ## Clean up cache and build artifacts
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=geodaedalus --cov-report=html --cov-report=term-missing

lint: ## Run linting
	ruff check geodaedalus tests
	ruff format --check geodaedalus tests

format: ## Format code
	ruff format geodaedalus tests
	ruff check --fix geodaedalus tests

type-check: ## Run type checking
	mypy geodaedalus

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

ci: lint type-check test ## Run all CI checks

docs: ## Build documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

build: ## Build package
	uv build

publish: build ## Publish to PyPI
	uv publish

# GeoDaedalus specific commands
search: ## Run a test search query
	geodaedalus search "Find igneous rock data from Hawaii with major elements" --dry-run

config-show: ## Show current configuration
	geodaedalus config --show

benchmark-list: ## List available benchmark datasets
	geo-bench list-datasets

benchmark-run: ## Run a quick benchmark test
	geo-bench run intent_understanding --max-samples 10

# Development database setup
setup-dev: dev-install ## Setup development environment
	cp env.example .env
	@echo "Please edit .env with your API keys"

# Docker commands (if needed)
docker-build: ## Build Docker image
	docker build -t geodaedalus .

docker-run: ## Run Docker container
	docker run -it --rm geodaedalus

# Metrics and monitoring
metrics-export: ## Export latest metrics
	geodaedalus export-metrics --format json

# Clean and reset everything
reset: clean ## Reset environment
	rm -rf .venv/
	rm -f .env 