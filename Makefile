.PHONY: fetch train predict pipeline frontend

fetch: ## Fetch raw market data
	poetry run python -m backend.workflows.fetch_data

train: ## Tune hyperparameters & train model
	poetry run python -m backend.workflows.train

predict: ## Run predictions
	poetry run python -m backend.workflows.predict

pipeline: fetch train predict ## Run full pipeline

frontend: ## Launch Streamlit dashboard
	poetry run streamlit run frontend/app.py
