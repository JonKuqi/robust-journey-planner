from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time

from src.routing.robust_journey_planner import RobustJourneyPlanner
from src.data.delay_data_handler import DelayDataHandler
from src.config.settings import get_settings

class EmpiricalCalibration:
    """Evaluates the reliability of the robust planner against historical actuals."""

    def __init__(
        self,
        robust_planner: RobustJourneyPlanner,
        test_start_date: str = "2026-01-01",
        test_end_date: str = "2026-01-31",
    ):
        self.planner = robust_planner
        self.settings = robust_planner.settings
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.spark = DelayDataHandler(settings=self.settings).get_spark()

    def fetch_historical_test_journeys(self, max_samples: int = 100) -> pd.DataFrame:
        """Fetch O-D pairs and arrival deadlines from historical hold-out data."""
        from pyspark.sql import functions as F
        
        # We look for successful trips in the hold-out range to use as test queries
        data_handler = DelayDataHandler(spark=self.spark, settings=self.settings)
        raw_df = data_handler.build_raw_delay_data(
            start_date=self.test_start_date,
            end_date=self.test_end_date,
            restrict_to_scope_stops=True
        )
        
        # Sample unique trip endpoints to serve as "Ground Truth" queries
        # We want trips that have a clear arrival time and actual arrival
        test_queries = (
            raw_df.select(
                "bpuic", "stop_name", "operating_day", "arr_time_ts", "arr_actual_ts", "trip_id"
            )
            .sample(withReplacement=False, fraction=1.0) # We limit later
            .limit(max_samples * 5) # Get a pool
        ).toPandas()

        # To simplify, we'll use these as destination points. 
        # For origins, we'll just pick a point ~30-60 mins before from the same trip or nearby.
        # But a more robust way is to just use these as "Arrival at Destination" queries.
        return test_queries.sample(n=min(len(test_queries), max_samples))

    def run_calibration_study(
        self, 
        quantiles: List[float] = [0.5, 0.7, 0.8, 0.9, 0.95],
        max_samples: int = 50
    ) -> Dict[float, float]:
        """Measure Success Rate for each quantile level."""
        test_data = self.fetch_historical_test_journeys(max_samples=max_samples)
        results = {}

        for q in quantiles:
            success_count = 0
            total_valid = 0
            
            print(f"Evaluating Calibration for Q={q}...")
            for _, row in test_data.iterrows():
                # Query: Get to row['bpuic'] by row['arr_time_ts']
                # Origin: For this study, we simulate a request from a nearby stop or earlier in the trip
                # To keep it simple and focused on the "Confidence" part, we check if our 
                # predicted arrival for a planned route matches reality.
                
                # In a real study, we'd pick a random origin. 
                # Here we just want to see if the 'robust_arrival' predicted by the model
                # at confidence Q is actually >= arr_actual in reality.
                
                # Let's simulate a plan for a single leg for simplicity in this baseline,
                # or a full plan if the planner is fast enough.
                
                # For calibration of the *model* itself:
                # We can just look at the legs of the test data.
                pass 

            # Refined logic: 
            # 1. Take a planned route.
            # 2. For each leg, get the actual delay from historical data.
            # 3. Check if the route succeeds.
            
        return results

    def run_simple_leg_calibration(
        self, 
        quantiles: List[float] = [0.5, 0.7, 0.8, 0.9, 0.95],
        max_samples: int = 100
    ) -> Dict[float, float]:
        """A faster version that evaluates individual legs to validate the DelayModel's Q-accuracy."""
        test_data = self.fetch_historical_test_journeys(max_samples=max_samples)
        results = {}

        for q in quantiles:
            successes = 0
            for _, row in test_data.iterrows():
                # Construct a dummy route step
                step = {
                    "type": "ride",
                    "to_stop": row['bpuic'],
                    "arrival_secs": row['arr_time_ts'].hour * 3600 + row['arr_time_ts'].minute * 60 + row['arr_time_ts'].second,
                    "travel_date": row['operating_day'],
                    "day_of_week": row['operating_day'].weekday() + 1, # Adjust to Spark 1=Sun? No, handled in model
                }
                
                # Annotate with delay
                annotated = self.planner.delay_model.add_delays_to_route({"steps": [step], "travel_date": row['operating_day']}, q=q)
                pred_delay = annotated['steps'][0]['predicted_delay_sec']
                
                actual_delay = (row['arr_actual_ts'] - row['arr_time_ts']).total_seconds()
                
                if pred_delay >= actual_delay:
                    successes += 1
            
            results[q] = successes / len(test_data) if len(test_data) > 0 else 0
            
        return results
