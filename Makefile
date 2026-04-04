.PHONY: help install install-dev setup frontend-install run-api run-frontend run \
	test test-unit lint format download-data train-models migrate-db docker-up docker-down clean

.DEFAULT_GOAL := help

PYTHON := python
PIP := pip

help:
	@echo "Available targets:"
	@echo "  install          Install runtime Python dependencies"
	@echo "  install-dev      Install the broader development environment"
	@echo "  setup            Copy .env.example to .env and install runtime dependencies"
	@echo "  frontend-install Install frontend dependencies"
	@echo "  run-api          Start the FastAPI backend"
	@echo "  run-frontend     Start the Vite frontend"
	@echo "  run              Alias for run-frontend"
	@echo "  test             Run the full pytest suite"
	@echo "  test-unit        Run unit tests only"
	@echo "  lint             Run flake8 and mypy"
	@echo "  format           Run black and isort"
	@echo "  download-data    Download daily market data"
	@echo "  train-models     Train local model artifacts"
	@echo "  migrate-db       Migrate SQLite data to PostgreSQL/TimescaleDB"
	@echo "  docker-up        Start the Docker stack"
	@echo "  docker-down      Stop the Docker stack"
	@echo "  clean            Remove local cache and coverage artifacts"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements_working.txt

setup:
	copy .env.example .env
	$(MAKE) install

frontend-install:
	cd quantvision && npm install

run-api:
	$(PYTHON) -m uvicorn src.api.main:app --reload --port 8000

run-frontend:
	cd quantvision && npm run dev

run:
	cd quantvision && npm run dev

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

lint:
	flake8 src tests scripts test_setup.py test_indicators.py
	mypy src

format:
	black src tests scripts test_setup.py test_indicators.py
	isort src tests scripts test_setup.py test_indicators.py

download-data:
	$(PYTHON) scripts/download_daily_data.py

train-models:
	$(PYTHON) scripts/train_models.py

migrate-db:
	$(PYTHON) scripts/migrate_sqlite_to_postgres.py

docker-up:
	docker compose -f docker/docker-compose.yml up --build

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist .mypy_cache rmdir /s /q .mypy_cache
	@if exist htmlcov rmdir /s /q htmlcov
	@echo "Cleaned local cache files"
