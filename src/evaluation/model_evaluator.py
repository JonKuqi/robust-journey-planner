from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from src.models.delay_model import DelayModel
from src.data.delay_data_handler import DelayDataHandler
from .metrics import rmse, mae, bias, pinball_loss

class ModelEvaluator:
    """Evaluates the DelayModel regression and quantile performance."""

    def __init__(self, delay_model: DelayModel, spark=None):
        self.model = delay_model
        self.settings = delay_model.metadata # Or pass settings explicitly
        self.spark = spark or delay_model.spark

    def evaluate_on_test_set(
        self, 
        test_start_date: str = "2026-01-01",
        test_end_date: str = "2026-01-31",
        max_samples: int = 5000
    ) -> Dict[str, Any]:
        """Run standard regression and quantile metrics on hold-out data."""
        data_handler = DelayDataHandler(spark=self.spark, settings=None) # Settings will be fetched from env if None
        
        # Build test data using the data handler
        # We reuse the feature building logic but on the hold-out range
        test_df, categorical_cols, numeric_cols = data_handler.build_feature_input_data(
            start_date=test_start_date,
            end_date=test_end_date,
            include_weather=True,
            include_calendar=True
        )
        
        # Sample for evaluation performance
        test_pd = test_df.limit(max_samples).toPandas()
        
        if test_pd.empty:
            return {"error": "No data found in the specified test range."}

        # Prepare features for prediction
        feature_rows = test_pd.to_dict('records')
        
        # We use the internal batch prediction method of the DelayModel
        # Note: _predict_all_quantiles_batch expects feature_rows as a list of dicts
        all_preds = self.model._predict_all_quantiles_batch(feature_rows)
        
        y_true = test_pd["label_delay_min"].values * 60.0 # Convert min to sec
        
        results = {
            "regression": {},
            "quantiles": {}
        }
        
        # Regression metrics (using p50 as the point estimate)
        if 0.5 in all_preds:
            y_p50 = np.array(all_preds[0.5])
            results["regression"] = {
                "rmse": rmse(y_true, y_p50),
                "mae": mae(y_true, y_p50),
                "bias": bias(y_true, y_p50)
            }
            
        # Pinball loss for all available quantiles
        for q, y_pred in all_preds.items():
            results["quantiles"][q] = {
                "pinball_loss": pinball_loss(y_true, np.array(y_pred), q)
            }
            
        return results
