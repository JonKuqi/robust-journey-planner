# +
import numpy as np
import pandas as pd

class ParetoFrontierEvaluator:
    def __init__(self, planner):
        self.planner = planner

    def evaluate(self, test_data, quantiles=[0.5, 0.7, 0.8, 0.9, 0.95]):
        pareto_results = []
        calibration_results = {}

        for q in quantiles:
            durations = []
            successes = 0
            
            for _, row in test_data.iterrows():
                step = {
                    "type": "ride", 
                    "to_stop": row['bpuic'], 
                    "line_text": row.get('line_text', 'UNKNOWN'),
                    "transport": row.get('transport', 'UNKNOWN'), 
                    "arrival_secs": row['arr_time_ts'].hour * 3600 + row['arr_time_ts'].minute * 60,
                    "travel_date": row['operating_day'], 
                    "day_of_week": row['operating_day'].weekday() + 1
                }
                
                annotated = self.planner.delay_model.add_delays_to_route(
                    {"steps": [step], "travel_date": row['operating_day']}, q=q
                )
                
                pred_delay = annotated['steps'][0]['predicted_delay_sec']
                actual_delay = (row['arr_actual_ts'] - row['arr_time_ts']).total_seconds()
                
                durations.append(pred_delay / 60.0)
                if pred_delay >= actual_delay: 
                    successes += 1
                    
            success_rate = successes / len(test_data) if len(test_data) > 0 else 0
            calibration_results[q] = success_rate
            pareto_results.append({
                "quantile": q, 
                "extra_time_min": np.mean(durations), 
                "success_rate": success_rate
            })

        return pd.DataFrame(pareto_results), calibration_results
# -


