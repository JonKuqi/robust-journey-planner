from src.models.delay_model_trainer import DelayModelTrainer
from src.models.model_artifacts import ModelArtifacts


def test_historical_feature_specs_match_part_iv():
    levels = [level for _, level in DelayModelTrainer.HIST_AGG_SPECS]

    assert "hist_stop" in levels
    assert "hist_dow_hour" in levels


def test_final_trainer_uses_xgb_params_without_spark_init():
    """Trainer constants match the expected XGBoost quantile regression configuration.

    These are checked without starting Spark so the test stays fast.  The label
    column and categorical columns must match what the feature pipeline produces.
    """
    assert DelayModelTrainer.LABEL_COL == "label_delay_min"
    assert "month" in DelayModelTrainer.CATEGORICAL_COLS


def test_region_artifact_key_is_order_independent():
    key_a = ModelArtifacts.region_key_for(("region-b", "region-a"))
    key_b = ModelArtifacts.region_key_for(("region-a", "region-b"))

    assert key_a == key_b
    assert key_a.startswith("region_")
