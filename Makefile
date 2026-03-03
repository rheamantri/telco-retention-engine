.PHONY: help setup data train timing retention metrics shap figures app all

help:
	@echo ""
	@echo "Targets:"
	@echo "  make setup     - install deps"
	@echo "  make data      - prepare/engineer dataset"
	@echo "  make train     - train churn model (xgboost pipeline + plots)"
	@echo "  make timing    - train churn timing models (30/60/90)"
	@echo "  make retention - build retention table v2 (risk + clv + actions)"
	@echo "  make metrics   - model metrics report"
	@echo "  make shap      - global SHAP + reason codes artifacts"
	@echo "  make figures   - build all figures gallery"
	@echo "  make app       - launch streamlit dashboard"
	@echo "  make all       - run full pipeline end to end"
	@echo ""

setup:
	pip install -r requirements.txt

data:
	python -m v2_upgrade.scripts.01_prepare_data

train:
	python -m v2_upgrade.scripts.02_train_churn_model_v1style

timing:
	python -m v2_upgrade.scripts.04_train_churn_timing

retention:
	python -m v2_upgrade.scripts.06_build_retention_table_v2

metrics:
	python -m v2_upgrade.scripts.07_model_metrics

shap:
	python -m v2_upgrade.scripts.03_reason_codes
	python -m v2_upgrade.scripts.08_global_shap

figures:
	python -m v2_upgrade.scripts.09_make_figures

app:
	streamlit run v2_upgrade/app/retention_dashboard.py

all: data train timing retention metrics shap figures
	@echo ""
	@echo "DONE. Now run: make app"
	@echo ""
