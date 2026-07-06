# +
import pandas as pd
import xgboost as xgb
from src.models.delay_model_trainer import DelayModelTrainer
from src.data.delay_data_handler import DelayDataHandler

def apply_training_overrides():
    """
    Applies class-level overrides to force the system into Numeric-Only mode 
    WITHOUT altering the original source files. Run this before initializing the trainer.
    """
    # 1. Force the Trainer to ignore all categorical definitions
    DelayModelTrainer.CATEGORICAL_COLS = [] 

    # 2. Patch the Data Handler to move categoricals to numerics
    original_build_features = DelayDataHandler.build_feature_input_data
    
    def fixed_build_features(self, *args, **kwargs):
        df, cats, nums = original_build_features(self, *args, **kwargs)
        return df, [], (cats + nums)
        
    DelayDataHandler.build_feature_input_data = fixed_build_features

    # 3. Patch the Infrastructure Bypass
    def patched_build_scope(self, *args, **kwargs):
        from pyspark.sql import functions as F
        return self.get_spark().table(self.settings.istdaten_table).select(F.col("bpuic").cast("string")).distinct()
        
    DelayDataHandler.build_scope_stops = patched_build_scope


def apply_inference_overrides(planner):
    """
    Applies instance-level overrides to the planner's delay model to handle 
    numeric-only batch prediction. Run this AFTER the planner is prepared.
    """
    def numeric_only_predict_batch(self, feature_rows):
        df = pd.DataFrame(feature_rows)
        results = {}
        for q_level, booster in self.xgb_models.items():
            f_names = booster.feature_names
            
            X = pd.DataFrame()
            for name in f_names:
                if name not in df.columns:
                    df[name] = 0.0
                # Force everything to float32 Number
                X[name] = pd.to_numeric(df[name], errors="coerce").fillna(0.0).astype("float32")
            
            dinf = xgb.DMatrix(X, enable_categorical=False)
            preds = booster.predict(dinf)
            results[q_level] = [max(0.0, float(p) * 60.0) for p in preds]
            
        return results

    # Dynamically bind the custom method to the existing model instance
    planner.delay_model._predict_all_quantiles_batch = numeric_only_predict_batch.__get__(planner.delay_model)
# -


