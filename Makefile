.PHONY: fetch train predict pipeline

fetch: ## Fetch raw market data
	poetry run python -m backend.workflows.fetch_data

train: ## Tune hyperparameters & train model
	poetry run python -m backend.workflows.train

predict: ## Run predictions
	poetry run python -m backend.workflows.predict

pipeline: fetch train predict ## Run full pipeline
