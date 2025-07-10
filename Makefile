# CloudyBot Development Makefile

.PHONY: help install install-dev test test-cov lint format type-check security clean run docs docker pre-commit

# Variables
PYTHON := python3
PIP := pip
PACKAGE := cloudybot
TEST_PATH := tests
SOURCE_PATH := cloudybot

# Default target
help: ## Show this help message
	@echo "CloudyBot Development Commands:"
	@echo "==============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install the package for production
	$(PIP) install -e .

install-dev: ## Install the package with development dependencies
	$(PIP) install -e ".[dev,test,docs]"
	pre-commit install

# Testing
test: ## Run tests
	pytest $(TEST_PATH) -v

test-cov: ## Run tests with coverage report
	pytest $(TEST_PATH) -v --cov=$(PACKAGE) --cov-report=term-missing --cov-report=html

test-quick: ## Run tests quickly (no coverage)
	pytest $(TEST_PATH) -x -v

# Code Quality
lint: ## Run all linting checks
	flake8 $(SOURCE_PATH) $(TEST_PATH)
	bandit -r $(SOURCE_PATH)
	safety check

format: ## Format code with black and isort
	black $(SOURCE_PATH) $(TEST_PATH)
	isort $(SOURCE_PATH) $(TEST_PATH)

format-check: ## Check if code is formatted correctly
	black --check $(SOURCE_PATH) $(TEST_PATH)
	isort --check-only $(SOURCE_PATH) $(TEST_PATH)

type-check: ## Run type checking with mypy
	mypy $(SOURCE_PATH)

security: ## Run security checks
	bandit -r $(SOURCE_PATH)
	safety check

# Pre-commit
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

# Development
run: ## Run the Streamlit app
	streamlit run app.py

run-debug: ## Run the Streamlit app in debug mode
	DEBUG=true streamlit run app.py --logger.level=debug

# Documentation
docs-serve: ## Serve documentation locally
	mkdocs serve

docs-build: ## Build documentation
	mkdocs build

docs-deploy: ## Deploy documentation to GitHub Pages
	mkdocs gh-deploy

# Cleaning
clean: ## Clean up build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf site/

clean-logs: ## Clean log files
	rm -rf logs/
	mkdir -p logs

# Docker
docker-build: ## Build Docker image
	docker build -t cloudybot:latest .

docker-run: ## Run Docker container
	docker run -p 8501:8501 --env-file .env cloudybot:latest

docker-dev: ## Run Docker container with volume mount for development
	docker run -p 8501:8501 -v $(PWD):/app --env-file .env cloudybot:latest

# Package management
requirements: ## Generate requirements.txt from pyproject.toml
	$(PIP) freeze > requirements-freeze.txt

update-deps: ## Update all dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev,test,docs]"

# Deployment
build: ## Build package for distribution
	$(PYTHON) -m build

upload-test: ## Upload to TestPyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

upload: ## Upload to PyPI
	$(PYTHON) -m twine upload dist/*

# Environment
env-setup: ## Set up development environment
	$(PYTHON) -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "source venv/bin/activate  # Linux/Mac"
	@echo "venv\\Scripts\\activate     # Windows"

env-check: ## Check environment and dependencies
	@echo "Python version:"
	@$(PYTHON) --version
	@echo "\nPip version:"
	@$(PIP) --version
	@echo "\nInstalled packages:"
	@$(PIP) list
	@echo "\nGit status:"
	@git status --porcelain

# CI/CD helpers
ci-install: ## Install dependencies for CI
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,test]"

ci-test: ## Run full test suite for CI
	pytest $(TEST_PATH) -v --cov=$(PACKAGE) --cov-report=xml --cov-report=term

ci-lint: ## Run all linting for CI
	black --check $(SOURCE_PATH) $(TEST_PATH)
	isort --check-only $(SOURCE_PATH) $(TEST_PATH)
	flake8 $(SOURCE_PATH) $(TEST_PATH)
	mypy $(SOURCE_PATH)
	bandit -r $(SOURCE_PATH)

# Development utilities
logs: ## View application logs
	tail -f logs/cloudybot.log

logs-error: ## View error logs only
	grep ERROR logs/cloudybot.log | tail -20

setup-git: ## Set up git hooks and configuration
	git config core.autocrlf input
	git config core.filemode false
	pre-commit install
	@echo "Git configuration updated and pre-commit hooks installed"

version: ## Show version information
	@echo "CloudyBot Development Environment"
	@echo "================================"
	@$(PYTHON) -c "from cloudybot import __version__; print(f'Version: {__version__}')"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version | cut -d' ' -f1-2)"
	@echo "Git: $$(git --version)"

# Quick development workflow
dev-setup: env-setup install-dev pre-commit-install ## Complete development setup
	@echo "Development environment setup complete!"
	@echo "You can now run 'make run' to start the application."

dev-check: format-check type-check lint test ## Run all development checks
	@echo "All development checks passed! ✅"

dev-fix: format lint ## Fix common code issues
	@echo "Code issues fixed! 🔧" 