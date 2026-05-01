# ==============================================================================
# Makefile — Smart Electricity Grid Anomaly Detection System
# Convenience targets for development, Docker, and operations.
# ==============================================================================

.PHONY: help up down restart logs build clean sim-live sim-ff venv install lint

# Default target — show available commands.
help: ## Show this help message
	@echo.
	@echo   Smart Grid Anomaly Detection — Make Targets
	@echo   ============================================
	@echo.
	@echo   Docker:
	@echo     make up           Start all services (detached)
	@echo     make down         Stop all services and remove containers
	@echo     make restart      Restart all services
	@echo     make build        Rebuild Docker images (no cache)
	@echo     make logs         Tail logs from all services
	@echo     make logs-sim     Tail simulator logs only
	@echo     make logs-edge    Tail edge processor logs only
	@echo.
	@echo   Local Development:
	@echo     make venv         Create Python virtual environment
	@echo     make install      Install simulator dependencies locally
	@echo     make sim-live     Run simulator in live MQTT mode (local)
	@echo     make sim-ff       Run fast-forward dataset generation (local)
	@echo.
	@echo   Maintenance:
	@echo     make clean        Remove containers, volumes, and caches
	@echo.

# ---- Docker Targets ----

up: ## Start all services in detached mode
	docker-compose up -d

down: ## Stop and remove all containers
	docker-compose down

restart: ## Restart all services
	docker-compose down
	docker-compose up -d

build: ## Rebuild all Docker images without cache
	docker-compose build --no-cache

logs: ## Tail logs from all services
	docker-compose logs -f

logs-sim: ## Tail simulator logs only
	docker-compose logs -f simulator

logs-edge: ## Tail edge processor logs only
	docker-compose logs -f edge_processor

# ---- Local Development Targets ----

venv: ## Create a Python virtual environment
	python -m venv .venv
	@echo Virtual environment created. Activate with:
	@echo   Windows:  .venv\Scripts\activate
	@echo   Linux:    source .venv/bin/activate

install: ## Install simulator dependencies into active venv
	pip install -r simulator/requirements.txt

sim-live: ## Run the simulator in live MQTT streaming mode
	python -m simulator.cli_entrypoint --mode live

sim-ff: ## Run fast-forward dataset generation (7 days, 50 meters)
	python -m simulator.cli_entrypoint --mode fastforward --days 7

# ---- Maintenance ----

clean: ## Remove containers, volumes, and Python caches
	docker-compose down -v --remove-orphans
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo Cleanup complete.
