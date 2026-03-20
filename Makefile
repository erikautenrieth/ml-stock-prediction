.PHONY: fetch train predict pipeline frontend

fetch: ## Fetch raw market data
	poetry run python -m backend.workflows.fetch_data

train: ## Train default model (ExtraTrees)
	poetry run python -m backend.workflows.train

train-lgbm: ## Train LightGBM model
	poetry run python -m backend.workflows.train --model lightgbm

train-model: ## Train specific model (usage: make train-model MODEL=lightgbm)
	poetry run python -m backend.workflows.train --model $(MODEL)

train-force: ## Train & register model regardless of performance
	poetry run python -m backend.workflows.train --force

train-experiment: ## Train in isolated experiment (usage: make train-experiment EXPERIMENT=my_test)
	poetry run python -m backend.workflows.train --experiment $(EXPERIMENT)

predict: ## Run predictions
	poetry run python -m backend.workflows.predict

pipeline: fetch train predict ## Run full pipeline

fe: ## Launch Streamlit dashboard
	poetry run streamlit run frontend/app.py
