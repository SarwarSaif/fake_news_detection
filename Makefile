# Makefile for Fake News Detection Project

PYTHON := python3
DATASET ?= MMFakeBench_val
LABEL ?= Fake
SIMILARITY ?= cosine
SCORE_MIN ?=
SCORE_MAX ?=

# -----------------------------
# Phony targets
# -----------------------------
.PHONY: run run-cache test outliers lint clean run-outliers

# -----------------------------
# Run the main experiment
# -----------------------------
run:
	$(PYTHON) -m src.main

run-cache:
	$(PYTHON) -m src.main --clear_cache

# -----------------------------
# Run tests
# -----------------------------
test:
	$(PYTHON) -m pytest -q

# -----------------------------
# Run Fact Checker
# -----------------------------
fact-check:
	$(PYTHON) -m src.models.fact_checker \
		--dataset MMFakeBench_val \
		--label Fake \
		--max_samples 50

# -----------------------------
# Show outliers (hardcoded)
# -----------------------------
outliers:
	$(PYTHON) -m src.utils.visualize_outliers \
		--dataset $(DATASET) \
		--label $(LABEL) \
		--similarity $(SIMILARITY) \
		$(if $(SCORE_MIN),--score_min $(SCORE_MIN)) \
		$(if $(SCORE_MAX),--score_max $(SCORE_MAX)) \
		--clean 

# -----------------------------
# Run outliers with custom args from command line
# Example: make run-outliers DATASET=GossipCop_val LABEL=fake SCORE_MAX=0.17 SCORE_MIN=0.8
# -----------------------------
run-outliers: outliers

# -----------------------------
# Lint code
# -----------------------------
lint:
	ruff check src

# -----------------------------
# Clean temporary/cache files
# -----------------------------
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache

# nohup python3 -m src.models.fact_checker --dataset GossipCop_val --label fake --max_samples 5 **&** 
# nohup python3 -m src.models.fact_checker --dataset MMFakeBench_val --label fake --max_samples 5 & 
# tail -f nohup.out
# pkill -f "tail -f nohup.out"
