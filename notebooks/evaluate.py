#
# np-dl-assign-2
# Evaluation

import numpy as np
import pandas as pd
from mlflow.tracking.client import MlflowClient

# Extract training run metadata & metrics from mlflow as dataframe for easier analysis
def extract_run_meta_metrics(experiment, float_cols=[], filter_unfinished=True):
    client = MlflowClient()
    # pull runs for experiment 
    experiment_id = client.get_experiment_by_name(experiment).experiment_id
    run_infos = client.list_run_infos(experiment_id)
    runs = [client.get_run(run_info.run_id) for run_info in run_infos]
    
    # extract run metrics and metadata from runs
    run_metrics = []
    run_metas = []
    for run in runs:
        # extract metadata from metrics
        run_id = run.info.run_id
        run_status = run.info.status
    
        # filter out unfinished runs if specified
        if filter_unfinished and run_status != "FINISHED":
            continue
        
        time_ms = (run.info.end_time - run.info.start_time
                   if run.info.status == "FINISHED" else np.nan)
        meta = {
            "artifact_uri": run.info.artifact_uri,
            "time_ms": time_ms,
            "run_id": run_id,
            "status": run.info.status,
        }
        meta.update(run.data.params)
        meta.update(run.data.tags)
        meta.update(run.data.metrics)
        run_metas.append(meta)

        # metrics history have to be extracted separately as they have
        # a separate epoch dimension
        metric_names = run.data.metrics.keys()
        histories = [client.get_metric_history(run_id, name) for name in metric_names]
        for history in histories:
            for metric in history:
                metric_dict = {
                    "run_id": run_id,
                    "metric": metric.key,
                    "step": metric.step,
                    "value": metric.value,
                }
                run_metrics.append(metric_dict)
                run_metrics.append(metric_dict)
    meta_df = pd.DataFrame(run_metas).astype({
        col: "float64" for col in float_cols
    }).set_index("run_id")
    
    metrics_df =  pd.DataFrame(run_metrics)
    return meta_df, metrics_df
