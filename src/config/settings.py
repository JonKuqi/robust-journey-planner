from __future__ import annotations

"""Central settings for the robust journey-planning MVP.

The project runs in two environments:

* Trino is used for the CSA timetable tables.
* Spark is used for delay-data preparation and model-artifact creation.

Most values can be overridden with environment variables so the notebook can be
reused by different group members without editing source files.
"""

from dataclasses import dataclass, field
import base64
import json
import os
import pwd
import time
from typing import Optional


DEFAULT_SHARED_SCHEMA = "iceberg.com490_iceberg"
DEFAULT_ISTDATEN_TABLE = "iceberg.sbb.istdaten"
DEFAULT_TIMETABLE_STOPS_TABLE = "iceberg.sbb.stops"
DEFAULT_GEO_SHAPES_TABLE = "iceberg.geo.shapes"
DEFAULT_GROUP_NAME = "J1"
DEFAULT_OPERATOR_IDS = ()  # empty = no filter, train on all Swiss operators


def _split_csv(value: Optional[str]) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _split_floats(value: Optional[str]) -> tuple[float, ...]:
    if not value:
        return ()
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _decode_token_subject(token: Optional[str]) -> Optional[str]:
    """Return the username stored in the EPFL JWT token, if available."""
    if not token:
        return None

    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        obj = json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None

    exp = obj.get("exp")
    if exp is not None and time.time() > int(exp) - 3600:
        raise RuntimeError("EPFL_COM490_TOKEN expires in less than one hour. Restart JupyterHub.")

    return obj.get("sub")


def infer_user_schema(token: Optional[str] = None) -> Optional[str]:
    """Infer the user Iceberg schema from the EPFL token."""
    username = _decode_token_subject(token or os.getenv("EPFL_COM490_TOKEN"))
    if not username:
        return None
    return f"iceberg.{username}_iceberg"


def default_artifacts_path() -> str:
    """Choose an artifact path that works locally and on the course cluster."""
    configured = os.getenv("COM490_ARTIFACTS_PATH")
    if configured:
        return configured
    if os.getenv("HADOOP_FS"):
        username = pwd.getpwuid(os.getuid()).pw_name
        return f"/user/{username}/robust_journey_planner/artifacts"
    return "artifacts"


@dataclass(frozen=True)
class ProjectSettings:
    """Runtime settings shared by data handlers, models, and planners."""

    group_name: str = field(default_factory=lambda: os.getenv("COM490_GROUP_NAME", DEFAULT_GROUP_NAME))
    shared_schema: str = field(default_factory=lambda: os.getenv("COM490_SHARED_SCHEMA", DEFAULT_SHARED_SCHEMA))
    user_schema: Optional[str] = field(
        default_factory=lambda: os.getenv("COM490_USER_SCHEMA") or infer_user_schema()
    )
    trino_url: Optional[str] = field(default_factory=lambda: os.getenv("TRINO_URL"))
    trino_token: Optional[str] = field(default_factory=lambda: os.getenv("EPFL_COM490_TOKEN"))
    hadoop_fs: Optional[str] = field(default_factory=lambda: os.getenv("HADOOP_FS"))

    istdaten_table: str = field(default_factory=lambda: os.getenv("COM490_ISTDATEN_TABLE", DEFAULT_ISTDATEN_TABLE))
    spark_user_istdaten_table: str = field(
        default_factory=lambda: os.getenv("COM490_SPARK_USER_ISTDATEN_TABLE", "default.istdaten")
    )
    istdaten_source_url: str = field(default_factory=lambda: os.getenv(
        "COM490_ISTDATEN_SOURCE_URL", "https://data.opentransportdata.swiss/dataset/istdaten"
    ))
    timetable_stops_table: str = field(
        default_factory=lambda: os.getenv("COM490_TIMETABLE_STOPS_TABLE", DEFAULT_TIMETABLE_STOPS_TABLE)
    )
    geo_shapes_table: str = field(default_factory=lambda: os.getenv("COM490_GEO_SHAPES_TABLE", DEFAULT_GEO_SHAPES_TABLE))
    model_start_date: str = field(default_factory=lambda: os.getenv("COM490_MODEL_START_DATE", "2024-10-01"))
    model_end_date: str = field(default_factory=lambda: os.getenv("COM490_MODEL_END_DATE", "2026-01-31"))
    final_train_start_date: str = field(default_factory=lambda: os.getenv("COM490_FINAL_TRAIN_START_DATE", "2024-10-01"))
    final_train_end_date: str = field(default_factory=lambda: os.getenv("COM490_FINAL_TRAIN_END_DATE", "2025-07-31"))
    final_val_start_date: str = field(default_factory=lambda: os.getenv("COM490_FINAL_VAL_START_DATE", "2025-10-01"))
    final_val_end_date: str = field(default_factory=lambda: os.getenv("COM490_FINAL_VAL_END_DATE", "2026-01-31"))
    operator_ids: tuple[str, ...] = field(
        default_factory=lambda: _split_csv(os.getenv("COM490_OPERATOR_IDS")) or DEFAULT_OPERATOR_IDS
    )
    region_uuids: tuple[str, ...] = field(default_factory=lambda: _split_csv(os.getenv("COM490_REGION_UUIDS")))
    sample_fraction: float = field(default_factory=lambda: float(os.getenv("COM490_SAMPLE_FRACTION", "0.1")))
    sample_seed: int = field(default_factory=lambda: int(os.getenv("COM490_SAMPLE_SEED", "42")))

    max_walk_m: int = field(default_factory=lambda: int(os.getenv("COM490_MAX_WALK_M", "500")))
    walking_speed_m_per_min: float = field(
        default_factory=lambda: float(os.getenv("COM490_WALKING_SPEED_M_PER_MIN", "50.0"))
    )

    walking_transfer_base_sec: float = field(
        default_factory=lambda: float(os.getenv("COM490_WALKING_TRANSFER_BASE_SEC", "120"))
    )

    artifacts_base_path: str = field(default_factory=default_artifacts_path)
    local_artifacts_path: str = field(
        default_factory=lambda: os.getenv(
            "COM490_LOCAL_ARTIFACTS_PATH", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "artifacts")
        )
    )
    weather_history_path: Optional[str] = field(default_factory=lambda: os.getenv("COM490_WEATHER_HISTORY_PATH"))
    weather_stations_path: Optional[str] = field(default_factory=lambda: os.getenv("COM490_WEATHER_STATIONS_PATH"))
    xgb_num_round: int = field(default_factory=lambda: int(os.getenv("COM490_XGB_NUM_ROUND", "100")))
    xgb_num_workers: int = field(default_factory=lambda: int(os.getenv("COM490_XGB_NUM_WORKERS", "16")))
    xgb_max_depth: int = field(default_factory=lambda: int(os.getenv("COM490_XGB_MAX_DEPTH", "6")))
    xgb_eta: float = field(default_factory=lambda: float(os.getenv("COM490_XGB_ETA", "0.15")))
    xgb_subsample: float = field(default_factory=lambda: float(os.getenv("COM490_XGB_SUBSAMPLE", "0.8")))
    xgb_colsample_bytree: float = field(default_factory=lambda: float(os.getenv("COM490_XGB_COLSAMPLE_BYTREE", "0.8")))
    xgb_min_child_weight: int = field(default_factory=lambda: int(os.getenv("COM490_XGB_MIN_CHILD_WEIGHT", "20")))
    xgb_quantile_levels: tuple[float, ...] = field(
        default_factory=lambda: _split_floats(os.getenv("COM490_XGB_QUANTILE_LEVELS"))
        or (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
    )
    spark_master: Optional[str] = field(default_factory=lambda: os.getenv("SPARK_MASTER"))


def get_settings(**overrides) -> ProjectSettings:
    """Build settings, optionally overriding selected dataclass fields."""
    settings = ProjectSettings()
    if not overrides:
        return settings
    values = {**settings.__dict__, **overrides}
    return ProjectSettings(**values)
