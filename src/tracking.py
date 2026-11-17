"""Tiny MLflow helper. Tracking must never break training, so every error is swallowed.

View runs with:  mlflow ui --backend-store-uri sqlite:///mlflow.db     (then open localhost:5000)
"""
from pathlib import Path

from src.config import ROOT

EXPERIMENT = "hydraulic-agent"


def log_run(run_name, params=None, metrics=None, artifacts=(), history=None) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params({k: str(v) for k, v in params.items()})
            for step, (train_loss, val_loss) in enumerate(history or [], start=1):
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=step)
            if metrics:
                clean = {k: float(v) for k, v in metrics.items()
                         if isinstance(v, (int, float)) and v == v}  # drops NaN
                mlflow.log_metrics(clean)
            for a in artifacts:
                if Path(a).exists():
                    mlflow.log_artifact(str(a))
        print(f"[mlflow] logged run '{run_name}'")
    except Exception as exc:  # noqa: BLE001
        print(f"[mlflow] skipped: {exc}")