# Gold Price Predictor — Dev shortcuts
# Usage: make <target>
# Requires: bash setup_mac.sh (run once)

PYTHON   := .venv/bin/python
STREAM   := .venv/bin/streamlit
SCRIPTS  := scripts

.DEFAULT_GOAL := help

.PHONY: help setup run train clean lint

help:
	@echo ""
	@echo "  \033[33m🥇 Gold Price Predictor\033[0m"
	@echo ""
	@echo "  make setup      Install dependencies (run once)"
	@echo "  make run        Launch dashboard → http://localhost:8501"
	@echo "  make train      Train ML model with 5y of data (recommended)"
	@echo "  make clean      Remove cached model files"
	@echo ""

setup:
	bash setup_mac.sh

run:
	source .venv/bin/activate && \
	$(STREAM) run app.py \
		--server.headless false \
		--browser.gatherUsageStats false \
		--server.port 8501

train:
	source .venv/bin/activate && \
	$(PYTHON) $(SCRIPTS)/train_local.py --period 5y --horizon 1
	@echo ""
	@echo "  \033[32m✓ Model saved. Commit it:\033[0m"
	@echo "    git add src/models/saved/ && git commit -m 'update trained model'"

train-all:
	@for h in 1 2 3 5; do \
		echo "Training horizon=$$h days..."; \
		source .venv/bin/activate && $(PYTHON) $(SCRIPTS)/train_local.py --period 5y --horizon $$h; \
	done

clean:
	rm -f src/models/saved/*.joblib src/models/saved/*.json
	@echo "Cleared saved models"
