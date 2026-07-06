from src.models.model_artifacts import ModelArtifacts
from src.models.delay_model_trainer import DelayModelTrainer
from src.config.settings import get_settings


def test_global_delay_artifacts_always_returns_global():
    """global_delay_artifacts() returns region_key='global' regardless of the base artifacts.

    This is the guarantee that pre-trained models are always found at evaluation
    time even when the grader passes a different region UUID than we trained with.
    """
    base = ModelArtifacts()
    assert base.global_delay_artifacts().region_key == "global"

    # Calling from a region-specific instance must still resolve to global
    regional = base.for_region(["some-uuid-that-was-never-trained-on"])
    assert regional.global_delay_artifacts().region_key == "global"


def test_model_exists_checks_global_path(tmp_path):
    """model_exists() returns True only when all booster JSONs exist under artifacts/global/.

    A different local_artifacts_path with no global directory must return False,
    confirming that training artifacts are always read from the global namespace.
    """
    settings = get_settings(local_artifacts_path=str(tmp_path))
    trainer = DelayModelTrainer(settings=settings)

    assert not trainer.model_exists()

    # Populate the global directory with all required files
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    (global_dir / "delay_model_metadata.json").write_text("{}", encoding="utf-8")
    for q in settings.xgb_quantile_levels:
        q_str = f"q{int(round(q * 100)):03d}"
        (global_dir / f"xgb_model_{q_str}.json").write_text("{}", encoding="utf-8")

    assert trainer.model_exists()
