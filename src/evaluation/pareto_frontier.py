from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.routing.robust_journey_planner import RobustJourneyPlanner
from .calibration import EmpiricalCalibration

class ParetoFrontierEvaluator:
    """Computes the 'Paranoia Tax' - Extra Travel Time vs Reliability."""

    def __init__(self, planner: RobustJourneyPlanner, calibration_engine: EmpiricalCalibration):
        self.planner = planner
        self.calibration = calibration_engine

    def compute_frontier(
        self, 
        quantiles: List[float] = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95],
        max_queries: int = 50
    ) -> pd.DataFrame:
        """Calculate avg travel time and success rate for each quantile."""
        
        # We use the calibration engine to get test queries
        test_data = self.calibration.fetch_historical_test_journeys(max_samples=max_queries)
        
        results = []
        
        # Baseline: Average travel time at Q=0 (or Q=0.5 if 0 is not available)
        baseline_q = 0.5
        
        for q in quantiles:
            print(f"Computing Pareto metrics for Q={q}...")
            durations = []
            success_count = 0
            
            for _, row in test_data.iterrows():
                # For each query, we plan a route
                # In this simplified version for the Pareto curve, 
                # we measure the 'Robust Duration' (including predicted delays)
                # and cross-reference with actual success from calibration logic.
                
                # Mocking a planning call to simulate the trade-off:
                # Higher Q -> Larger delay buffer -> Longer 'Robust' Travel Time
                
                # In reality, we'd call self.planner.plan(...) here
                # But for the evaluation module, we focus on the relationship:
                
                # Let's assume we annotate a fixed path with different Qs
                # to see the 'extra time' added by the robust model.
                
                step = {
                    "type": "ride",
                    "to_stop": row['bpuic'],
                    "arrival_secs": 36000, # Constant for duration diff
                    "travel_date": row['operating_day'],
                    "day_of_week": 1,
                }
                
                annotated = self.planner.delay_model.add_delays_to_route({"steps": [step], "travel_date": row['operating_day']}, q=q)
                extra_sec = annotated['steps'][0]['predicted_delay_sec']
                
                actual_delay = (row['arr_actual_ts'] - row['arr_time_ts']).total_seconds()
                is_success = extra_sec >= actual_delay
                
                durations.append(extra_sec / 60.0) # In minutes
                if is_success:
                    success_count += 1
            
            results.append({
                "quantile": q,
                "extra_time_min": np.mean(durations),
                "success_rate": success_count / len(test_data) if len(test_data) > 0 else 0
            })
            
        return pd.DataFrame(results)
