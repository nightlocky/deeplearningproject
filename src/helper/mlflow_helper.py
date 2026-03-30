import mlflow
import mlflow.pytorch
import os

class MLFlowTracker:
    def __init__(self, experiment_name="OCT_Anomaly_Detection_Separated"):
        self.experiment_name = experiment_name
        mlruns_dir = "/workspace/mlruns"
        
        # Ensure the directory exists
        if not os.path.exists(mlruns_dir):
            os.makedirs(mlruns_dir, exist_ok=True)
            
        mlflow.set_tracking_uri(f"file:{mlruns_dir}")
        mlflow.set_experiment(self.experiment_name)
        self.active_run = None

    def start_run(self, run_name=None):
        self.active_run = mlflow.start_run(run_name=run_name)
        return self.active_run

    def end_run(self):
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics_dict, step=None):
            mlflow.log_metrics(metrics_dict, step=step)

    def log_artifact(self, local_path):
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path)
        else:
            print(f"Warning: Artifact path {local_path} does not exist.")

    def log_model(self, model, artifact_path="model"):
        mlflow.pytorch.log_model(model, artifact_path)

    def __enter__(self):
        self.start_run()   
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()