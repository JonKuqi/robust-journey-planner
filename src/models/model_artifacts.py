from __future__ import annotations

"""Paths for saved model artifacts."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Iterable

from src.config.settings import ProjectSettings, get_settings


@dataclass(frozen=True)
class ModelArtifacts:
    """Resolve artifact paths used by the delay predictor."""

    base_path: str | None = None
    settings: ProjectSettings | None = None
    region_key: str | None = None

    def __post_init__(self):
        settings = self.settings or get_settings()
        object.__setattr__(self, "settings", settings)
        if self.base_path is None:
            object.__setattr__(self, "base_path", settings.artifacts_base_path)
        if self.region_key is None:
            object.__setattr__(self, "region_key", "global")

    def for_region(self, region_uuids: Iterable[str] | None = None) -> "ModelArtifacts":
        """Return artifact paths namespaced by region UUIDs."""
        key = self.region_key_for(region_uuids)
        return ModelArtifacts(
            base_path=f"{self.base_path}/{key}",
            settings=self.settings,
            region_key=key,
        )

    def global_delay_artifacts(self) -> "ModelArtifacts":
        """Return artifacts under the fixed 'global' namespace.

        Delay models are trained on all available data (no region filter), so
        pre-trained artifacts are always found regardless of the region UUID
        passed at evaluation time. See MODEL_IDEA.md for rationale.
        """
        return self.for_region(None)

    def delay_training_data_path(self) -> str:
        """Path for cleaned delay-training records."""
        return f"{self.base_path}/delay_training_data.parquet"

    def delay_feature_data_path(self) -> str:
        """Path for final feature rows used by Spark ML training."""
        return f"{self.base_path}/delay_feature_input.parquet"

    def xgb_model_path(self, quantile: float) -> str:
        """Path for a trained XGBoost quantile PipelineModel."""
        q_str = f"q{int(round(quantile * 100)):03d}"
        return f"{self.base_path}/xgb_model_{q_str}"

    def local_xgb_model_json_path(self, quantile: float) -> str:
        """Local path for a driver-trained XGBoost booster saved as JSON."""
        q_str = f"q{int(round(quantile * 100)):03d}"
        return f"{self.settings.local_artifacts_path}/{self.region_key}/xgb_model_{q_str}.json"

    def hist_aggs_json_path(self) -> str:
        """Local JSON lookup of historical delay aggregates for inference."""
        override = os.getenv("COM490_HIST_AGGS_JSON_PATH")
        if override:
            return override
        return f"{self.settings.local_artifacts_path}/{self.region_key}/hist_aggs.json"

    def climatological_weather_json_path(self) -> str:
        """Local JSON of (station_id, month) climatological weather averages."""
        override = os.getenv("COM490_CLIM_WEATHER_JSON_PATH")
        if override:
            return override
        return f"{self.settings.local_artifacts_path}/{self.region_key}/climatological_weather.json"

    def precomputed_delays_json_path(self) -> str:
        """Local JSON lookup of pre-computed delay quantiles per (stop, line, hour, dow, month).

        Files live under a shared precomputed_delays/ folder, one per region key, so
        anyone who clones the repo gets per-region lookups without overwriting anyone
        else's artifacts.
        """
        override = os.getenv("COM490_PRECOMPUTED_DELAYS_JSON_PATH")
        if override:
            return override
        return f"{self.settings.local_artifacts_path}/precomputed_delays/{self.region_key}.json"

    def stop_station_mapping_json_path(self) -> str:
        """Local JSON mapping bpuic → nearest weather station id."""
        override = os.getenv("COM490_STOP_STATION_JSON_PATH")
        if override:
            return override
        return f"{self.settings.local_artifacts_path}/{self.region_key}/stop_station_mapping.json"

    def stop_canton_mapping_json_path(self) -> str:
        """Local JSON mapping bpuic → 2-letter Swiss canton code."""
        override = os.getenv("COM490_STOP_CANTON_JSON_PATH")
        if override:
            return override
        return f"{self.settings.local_artifacts_path}/{self.region_key}/stop_canton_mapping.json"

    def eval_results_json_path(self) -> str:
        """Local JSON with post-hoc evaluation results (pinball loss per quantile)."""
        override = os.getenv("COM490_EVAL_RESULTS_JSON_PATH")
        if override:
            return override
        return f"{self.settings.local_artifacts_path}/global/eval_results.json"

    def baseline_results_json_path(self) -> str:
        """Local JSON with baseline comparison results (pinball loss per model per quantile)."""
        return f"{self.settings.local_artifacts_path}/global/baseline_results.json"

    def train_loss_curves_json_path(self, timestamp_str: str) -> str:
        """Local JSON with per-round training loss per quantile for one training run."""
        return f"{self.settings.local_artifacts_path}/global/train_loss_curves_{timestamp_str}.json"

    def model_metadata_path(self) -> str:
        """Local JSON metadata needed to build model inference rows."""
        override = os.getenv("COM490_DELAY_MODEL_METADATA_PATH")
        if override:
            return override
        return f"{self.settings.local_artifacts_path}/{self.region_key}/delay_model_metadata.json"

    def ensure_local_base_dir(self) -> None:
        """Create the local artifact directory when using a local path."""
        if "://" in self.base_path or self.base_path.startswith("/user/"):
            return
        Path(self.base_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def region_key_for(region_uuids: Iterable[str] | None = None) -> str:
        """Build a stable artifact namespace for a set of region UUIDs."""
        regions = sorted(str(region).strip() for region in (region_uuids or ()) if str(region).strip())
        if not regions:
            return "global"
        digest = hashlib.sha1(",".join(regions).encode("utf-8")).hexdigest()[:12]
        return f"region_{digest}"
