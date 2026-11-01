"""
MLflow Experiment Tracking Module for NASA C-MAPSS Predictive Maintenance.
Logs hyperparameter configs, evaluation metrics (RMSE, MAE, NASA Score),
and model artifacts to an SQLite-backed MLflow tracking server.
"""

import os
from typing import Any, Dict, Optional
import mlflow

class MLflowTracker:
    """Wrapper around MLflow for standardized experiment logging."""
    def __init__(self, experiment_name: str = "CMAPSS_FD001_Predictive_Maintenance", tracking_uri: str = "sqlite:///mlflow.db"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"[Warning] Could not initialize MLflow URI '{self.tracking_uri}': {e}. Using default local tracking.")

    def log_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Log an experiment run to MLflow.
        Returns the active run_id.
        """
        try:
            with mlflow.start_run(run_name=run_name) as run:
                # Log tags
                if tags:
                    mlflow.set_tags(tags)
                    
                # Log hyperparameters
                for k, v in params.items():
                    mlflow.log_param(k, v)
                    
                # Log evaluation metrics
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v))
                    
                # Log local artifacts
                if artifacts:
                    for local_path, artifact_dir in artifacts.items():
                        if os.path.exists(local_path):
                            mlflow.log_artifact(local_path, artifact_dir)
                            
                return run.info.run_id
        except Exception as e:
            print(f"[Warning] MLflow logging failed: {e}")
            return None
