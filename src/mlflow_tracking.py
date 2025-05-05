import mlflow
import mlflow.sklearn
from src.train_pipeline import train_and_evaluate

def run_experiment():
    # Set experiment name
    mlflow.set_experiment("California_Housing_Regression")

    with mlflow.start_run():
        # Train model and get metrics
        pipeline, mse, r2 = train_and_evaluate()

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        # Log model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print("🔁 Experiment logged to MLflow.")

if __name__ == "__main__":
    run_experiment()