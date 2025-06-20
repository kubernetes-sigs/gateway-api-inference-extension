# Makefile for dynamic-lora-sidecar

PYTHON_VERSION := 3.10
VENV_DIR := venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: help venv install test clean

help: ## Show available targets
	@echo "Available targets:"
	@echo "  venv     - Create virtual environment"
	@echo "  install  - Install dependencies"
	@echo "  test     - Run unit tests"
	@echo "  clean    - Clean up virtual environment"

venv: $(VENV_DIR)/bin/activate ## Create virtual environment

$(VENV_DIR)/bin/activate:
	python$(PYTHON_VERSION) -m venv $(VENV_DIR)

install: venv ## Install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test: install ## Run unit tests
	$(PYTHON) -m unittest discover -v -s sidecar

clean: ## Clean up virtual environment
	rm -rf $(VENV_DIR)
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
