import mlflow
import pandas as pd
from sklearn.datasets import fetch_california_housing
from mlflow.tracking import MlflowClient

def load_latest_model_and_predict():
    experiment_name = "California_Housing_Regression"
    mlflow.set_experiment(experiment_name)
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"No experiment named '{experiment_name}' found.")
        return

    # Get runs for the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

    if runs.empty:
        print("❌ No runs found in this experiment.")
        return

    run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{run_id}/model"

    # Load model
    model = mlflow.sklearn.load_model(model_uri)

    # Sample data
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)

    # Predict
    predictions = model.predict(X.head())
    print("✅ Sample predictions on first 5 rows:")
    print(predictions)

if __name__ == "__main__":
    load_latest_model_and_predict()
