from  evidently import Report
from  evidently.metrics import DataDriftPreset
import pandas as pd

def check_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
    """
    Returns True if data drift is detected based on Evidently presets.
    """
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)
    
    report_dict = drift_report.as_dict()

    dataset_drifted = report_dict["metrics"][0]["result"]["dataset_drift"]
    
    return dataset_drifted