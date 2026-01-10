.PHONY: help install install-dev setup clean test lint format run

.DEFAULT_GOAL := help

PYTHON := python
PIP := pip
STREAMLIT := streamlit

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# ================================
# SETUP & INSTALLATION
# ================================
install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements-dev.txt

setup: ## Complete project setup
	copy .env.example .env
	$(MAKE) install-dev
	@echo "Setup complete! Edit .env file with your configuration."

# ================================
# DATA MANAGEMENT
# ================================
download-data: ## Download all stock data
	$(PYTHON) scripts/download_daily_data.py

update-data: ## Update data (daily task)
	$(PYTHON) scripts/update_data.py

# ================================
# MODEL TRAINING
# ================================
train-xgboost: ## Train XGBoost model
	$(PYTHON) scripts/train_model.py --model xgboost

train-lstm: ## Train LSTM model
	$(PYTHON) scripts/train_model.py --model lstm

train-all: ## Train all models
	$(PYTHON) scripts/train_all_models.py

# ================================
# DASHBOARD
# ================================
run: ## Run Streamlit dashboard
	$(STREAMLIT) run src/dashboard/app.py --server.port 8501

run-dev: ## Run dashboard in development mode
	$(STREAMLIT) run src/dashboard/app.py --server.port 8501 --server.runOnSave true

# ================================
# CODE QUALITY
# ================================
lint: ## Run linting checks
	flake8 src/ tests/
	mypy src/

format: ## Format code with Black
	black src/ tests/
	isort src/ tests/

test: ## Run all tests
	pytest tests/ -v --cov=src --cov-report=html

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

# ================================
# DOCKER
# ================================
docker-build: ## Build Docker image
	docker build -f docker/Dockerfile.prod -t stock-prediction-dashboard:latest .

docker-run: ## Run Docker container
	docker run -p 8501:8501 --env-file .env stock-prediction-dashboard:latest

# ================================
# UTILITIES
# ================================
clean: ## Clean up temporary files
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist .mypy_cache rmdir /s /q .mypy_cache
	@if exist htmlcov rmdir /s /q htmlcov
	@echo "Cleaned temporary files"

jupyter: ## Start Jupyter notebook
	jupyter notebook notebooks/
