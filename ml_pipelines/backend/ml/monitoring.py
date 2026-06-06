from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd

def check_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
    """
    Returns True if data drift is detected based on Evidently presets.
    Extracts the status directly from the lower-level metrics list.
    """
    # 1. Initialize and execute the evaluation report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)

    # 2. Extract from the compiled metrics tracking engine
    try:
        # Check the lower-level metrics list that Actually holds the ran data classes
        metrics_list = getattr(drift_report, '_first_level_metrics', []) or getattr(drift_report, 'metrics', [])
        
        for metric in metrics_list:
            # Safely pull the result mapping structure
            if hasattr(metric, 'get_result'):
                result = metric.get_result()
                # Check for the dataset-level drift boolean flag
                if hasattr(result, 'dataset_drift'):
                    return bool(result.dataset_drift)
                # Alternative attribute name in older version layers
                elif hasattr(result, 'metrics') and 'dataset_drift' in str(result):
                    return bool(getattr(result, 'dataset_drift', False))

        # 3. Last resort string-matching fallback (if all object lookups fail)
        # Even if dictionary conversion fails, the report string representation contains the flag
        report_str = str(drift_report)
        if "dataset_drift=True" in report_str.replace(" ", ""):
            return True
            
    except Exception as e:
        print(f"⚠️ All extraction vectors failed: {e}. Defaulting drift to False.")
        
    return False