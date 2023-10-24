import mlflow
import dvc.api
import pandas as pd


def yield_artifacts(run_id, path=None):
    """Yield all artifacts in the specified run"""
    client = mlflow.tracking.MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


def fetch_logged_data(run_id):
    """Fetch params, metrics, tags, and artifacts in the specified run"""
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }
    
def dvc_open(path, url, branch):
    with dvc.api.open(
        path = path,  ## 데이터 경로
        repo = url,  ## github repo 경로,
        rev =  branch ## 현재는 branch 기준
    ) as f:
        return pd.read_csv(f, sep=",")

def find_experiment_id(experiment_name):
    current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
    return current_experiment["experiment_id"]
