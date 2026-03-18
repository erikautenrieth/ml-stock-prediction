"""Run the full pipeline: fetch → train → log to DagsHub."""
from backend.workflows.fetch_data import fetch_and_store
from backend.workflows.train import train_model

print("=" * 60)
print("STEP 1: Fetching data & computing features")
print("=" * 60)
df, manifest = fetch_and_store()
print(f"Features ready: {manifest.row_count} rows, {len(manifest.columns)} cols")

print()
print("=" * 60)
print("STEP 2: Training model with Optuna tuning (10 trials)")
print("=" * 60)
result = train_model(do_tuning=True, n_trials=10)
print(f"Training complete!")
print(f"  Model:    {result.model_name}")
print(f"  Accuracy: {result.accuracy:.4f}")
print(f"  F1:       {result.f1:.4f}")
print(f"  Run ID:   {result.run_id}")
print(f"  URI:      {result.model_uri}")
