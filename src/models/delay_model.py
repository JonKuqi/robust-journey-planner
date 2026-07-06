from __future__ import annotations

"""XGBoost quantile delay model for route planning inference."""

from copy import deepcopy
import json
from pathlib import Path
from datetime import date, datetime
from typing import Mapping, Optional

from src.data.calendar_data import CalendarDataHandler
from src.data.weather_data import WeatherDataHandler
from .model_artifacts import ModelArtifacts


_BUS_TRANSPORT_CODES = frozenset({"BUS", "CAR", "EV", "KB", "B", "EXB", "BN", "BP", "RUB"})

# Spark dayofweek convention: 1=Sun, 2=Mon, ..., 7=Sat
_DAY_NAME_TO_DOW = {
    "sunday": 1, "monday": 2, "tuesday": 3, "wednesday": 4,
    "thursday": 5, "friday": 6, "saturday": 7,
}


class DelayModel:
    """Predict route-step delay quantiles using trained XGBoost models."""

    DEFAULT_QUANTILE_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)

    def __init__(
        self,
        xgb_models: dict | None = None,
        hist_aggs: dict | None = None,
        clim_weather: dict | None = None,
        stop_station: dict | None = None,
        stop_canton: dict | None = None,
        metadata: dict | None = None,
        precomputed_delays: dict | None = None,
        spark=None,
    ):
        self.xgb_models: dict[float, object] = xgb_models or {}
        self.hist_aggs: dict = hist_aggs or {}
        self.clim_weather: dict = clim_weather or {}
        self.stop_station: dict = stop_station or {}
        self.stop_canton: dict = stop_canton or {}
        self.metadata: dict = metadata or {}
        self.precomputed_delays: dict = precomputed_delays or {}
        self.spark = spark

        levels = self.metadata.get("quantile_levels", list(self.DEFAULT_QUANTILE_LEVELS))
        self.quantile_columns: dict[float, str] = {
            q: f"p{int(round(q * 100)):02d}_delay_sec" for q in levels
        }

    @classmethod
    def load(
        cls,
        artifacts: ModelArtifacts | None = None,
        region_artifacts: ModelArtifacts | None = None,
        spark=None,
    ) -> "DelayModel":
        """Load all XGBoost quantile models and supporting artifacts.

        Models, metadata, hist_aggs, weather, and canton mappings are always loaded
        from the global artifact namespace — training is region-independent.

        Precomputed delays are region-specific: if region_artifacts is provided its
        precomputed_delays_json_path() is used; otherwise the lookup is empty and
        all predictions fall back to live XGBoost inference.

        Spark is not required — all inference runs on the driver.
        """
        global_artifacts = (artifacts or ModelArtifacts()).global_delay_artifacts()
        metadata = cls._load_json(global_artifacts.model_metadata_path())
        quantile_levels = metadata.get("quantile_levels", list(cls.DEFAULT_QUANTILE_LEVELS))

        xgb_models: dict[float, object] = {}
        try:
            import xgboost as xgb
            for q in quantile_levels:
                path = global_artifacts.local_xgb_model_json_path(q)
                if Path(path).exists():
                    booster = xgb.Booster()
                    booster.load_model(path)
                    xgb_models[float(q)] = booster
        except ImportError as exc:
            raise RuntimeError("xgboost package is required for inference.") from exc

        if not xgb_models:
            raise RuntimeError(
                "No XGBoost model files found. Run DelayModelTrainer().train() first."
            )

        hist_aggs = cls._load_json(global_artifacts.hist_aggs_json_path())
        clim_weather = cls._load_json(global_artifacts.climatological_weather_json_path())
        stop_station = cls._load_json(global_artifacts.stop_station_mapping_json_path())
        stop_canton = cls._load_json(global_artifacts.stop_canton_mapping_json_path())

        precomputed_delays = (
            cls._load_json(region_artifacts.precomputed_delays_json_path())
            if region_artifacts is not None
            else {}
        )

        return cls(
            xgb_models=xgb_models,
            hist_aggs=hist_aggs,
            clim_weather=clim_weather,
            stop_station=stop_station,
            stop_canton=stop_canton,
            metadata=metadata,
            precomputed_delays=precomputed_delays,
            spark=spark,
        )

    def add_delays_to_route(self, route: Mapping[str, object], q: float = 0.90) -> dict:
        """Annotate ride steps with predicted quantile delays.

        Each ride step receives p50_delay_sec … p95_delay_sec (full distribution
        for ConfidenceEvaluator CDF interpolation) and predicted_delay_sec (the
        q-th quantile, for robust arrival display).
        """
        out = deepcopy(dict(route))
        travel_date = out.get("travel_date")
        day_of_week = out.get("day_of_week")
        month = self._month_from_route(out)

        steps = out.get("steps", [])
        ride_indices: list[int] = []
        feature_rows: list[dict] = []
        calendar_cache: dict = {}  # bpuic → calendar_features (stops in different cantons may differ)

        for i, step in enumerate(steps):
            if step.get("type") != "ride":
                step["predicted_delay_sec"] = 0.0
                continue
            bpuic = step.get("to_stop")
            if bpuic not in calendar_cache:
                calendar_cache[bpuic] = self._get_calendar_features(travel_date, bpuic)
            arrival_secs = step.get("arrival_secs")
            hour = int(arrival_secs // 3600 % 24) if isinstance(arrival_secs, (int, float)) else step.get("hour")
            feature_rows.append(self._build_feature_row(
                bpuic=bpuic,
                line_text=step.get("line_text"),
                transport=step.get("transport", ""),
                hour=hour,
                day_of_week=day_of_week,
                month=month,
                calendar_features=calendar_cache[bpuic],
            ))
            ride_indices.append(i)

        if not feature_rows:
            return out

        all_preds = self._predict_all_quantiles_batch(feature_rows)

        q_col = self._column_for_q(q)
        for j, step_idx in enumerate(ride_indices):
            step = steps[step_idx]
            for q_level, col in self.quantile_columns.items():
                preds = all_preds.get(q_level, [])
                step[col] = preds[j] if j < len(preds) else 0.0
            step["predicted_delay_sec"] = step.get(q_col, 0.0)
            step["robust_arrival_secs"] = float(step.get("arrival_secs", 0)) + step["predicted_delay_sec"]
            step["robust_arrival_time"] = self._secs_to_time(step["robust_arrival_secs"])

        return out

    def _predict_all_quantiles_batch(self, feature_rows: list[dict]) -> dict[float, list[float]]:
        """Return {quantile: [pred_sec, ...]} trying the pre-computed lookup first.

        For each feature row, builds a key from the categorical columns and looks it up
        in precomputed_delays. Rows not found in the lookup fall back to live XGBoost
        inference via _predict_xgb, so new stops or lines are handled automatically.
        """
        if not self.xgb_models:
            return {q: [0.0] * len(feature_rows) for q in self.quantile_columns}

        categorical_cols = self.metadata.get("categorical_cols", [])
        quantile_levels = sorted(self.xgb_models.keys())

        if not self.precomputed_delays or not categorical_cols:
            return self._predict_xgb(feature_rows)

        results: dict[float, list[float | None]] = {q: [None] * len(feature_rows) for q in quantile_levels}
        live_indices: list[int] = []
        live_rows: list[dict] = []

        for i, row in enumerate(feature_rows):
            key = "__".join(str(row.get(c, "")) for c in categorical_cols)
            preds = self.precomputed_delays.get(key)
            if preds is not None:
                for j, q in enumerate(quantile_levels):
                    results[q][i] = float(preds[j]) if j < len(preds) else 0.0
            else:
                live_indices.append(i)
                live_rows.append(row)

        if live_rows:
            live_preds = self._predict_xgb(live_rows)
            for j, orig_idx in enumerate(live_indices):
                for q in quantile_levels:
                    results[q][orig_idx] = live_preds.get(q, [0.0] * len(live_rows))[j]

        final = {q: [v if v is not None else 0.0 for v in results[q]] for q in quantile_levels}

        # Independently trained models can produce crossing quantiles (e.g. p85 < p80).
        # Sort per sample to enforce monotonicity before the ConfidenceEvaluator uses
        # these values for CDF interpolation.
        for i in range(len(feature_rows)):
            vals = sorted(final[q][i] for q in quantile_levels)
            for j, q in enumerate(quantile_levels):
                final[q][i] = vals[j]

        return final

    def _predict_xgb(self, feature_rows: list[dict]) -> dict[float, list[float]]:
        """Run all quantile XGBoost models on a batch; return {quantile: [pred_sec, ...]}.

        Replicates the StringIndexer encoding from training: each categorical value is
        mapped to its integer index using the saved vocabulary (frequency-ranked, same
        as Spark StringIndexer with handleInvalid="keep"). Features are assembled in the
        exact column order recorded in metadata["feature_cols"] by the trainer.
        """
        if not self.xgb_models:
            return {q: [0.0] * len(feature_rows) for q in self.quantile_columns}

        try:
            import pandas as pd
            import xgboost as xgb

            categorical_cols = self.metadata.get("categorical_cols", [])
            all_numeric_cols = self.metadata.get("all_numeric_cols", [])
            feature_cols = self.metadata.get("feature_cols", [])
            category_labels = self.metadata.get("category_labels", {})

            df = pd.DataFrame(feature_rows)

            for col in categorical_cols:
                vocab = category_labels.get(col, [])
                vocab_map = {v: float(i) for i, v in enumerate(vocab)}
                unknown_idx = float(len(vocab))
                raw = df[col].astype(str) if col in df.columns else pd.Series(["UNKNOWN"] * len(df))
                df[f"{col}_index"] = raw.map(lambda v, m=vocab_map, u=unknown_idx: m.get(v, u))

            for col in all_numeric_cols:
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

            valid_cols = [c for c in feature_cols if c in df.columns]
            dinf = xgb.DMatrix(df[valid_cols])

            results: dict[float, list[float]] = {}
            for q_level, booster in self.xgb_models.items():
                preds = booster.predict(dinf)
                results[q_level] = [max(0.0, float(p) * 60.0) for p in preds]
            return results
        except Exception:
            return {q: [0.0] * len(feature_rows) for q in self.quantile_columns}

    def _build_feature_row(
        self,
        bpuic,
        line_text,
        transport: str,
        hour,
        day_of_week,
        month,
        calendar_features: dict,
    ) -> dict:
        """Build a complete feature dict matching the training schema."""
        categorical_cols = self.metadata.get("categorical_cols", [])
        all_numeric_cols = self.metadata.get("all_numeric_cols", [])
        weather_bus_cols = self.metadata.get("weather_bus_cols", [])
        calendar_numeric_cols = self.metadata.get("calendar_numeric_cols", [])
        hist_numeric_cols = self.metadata.get("hist_numeric_cols", [])

        dow = self._norm_day(day_of_week)

        row: dict = {
            "bpuic": str(bpuic) if bpuic is not None else "UNKNOWN",
            "line_text": str(line_text).upper() if line_text is not None else "UNKNOWN",
            "hour": str(int(hour)) if hour is not None else "0",
            "day_of_week": str(dow) if dow is not None else "0",
            "month": str(int(month)) if month is not None else "0",
            "day_type": str(self._day_type_from_calendar(calendar_features)),
        }
        for c in categorical_cols:
            row.setdefault(c, "UNKNOWN")

        is_bus = str(transport).strip().upper() in _BUS_TRANSPORT_CODES
        weather = self._get_weather_features(bpuic=bpuic, month=month, is_bus=is_bus)
        for c in weather_bus_cols:
            row[c] = float(weather.get(c, 0.0))

        for c in calendar_numeric_cols:
            row[c] = float(calendar_features.get(c, 0.0))

        hist = self._get_hist_agg_features(bpuic=bpuic, line_text=line_text, hour=hour, day_of_week=dow, month=month)
        for c in hist_numeric_cols:
            row[c] = float(hist.get(c, 0.0))

        for c in all_numeric_cols:
            row.setdefault(c, 0.0)

        return row

    def _get_weather_features(self, bpuic, month, is_bus: bool) -> dict:
        """Return bus-specific weather features from climatological averages."""
        zero = {c: 0.0 for c in WeatherDataHandler.WEATHER_BUS_COLS}
        if not is_bus or not self.clim_weather or not self.stop_station:
            return zero
        station_id = self.stop_station.get(str(bpuic) if bpuic is not None else "")
        if station_id is None:
            return zero
        key = f"{station_id}__{int(month)}" if month is not None else None
        if key is None:
            return zero
        weather_row = self.clim_weather.get(key, {})
        return {
            f"{base}_bus": float(weather_row.get(base, 0.0) or 0.0)
            for base in WeatherDataHandler.WEATHER_NUMERIC_COLS
        }

    def _get_hist_agg_features(self, bpuic, line_text, hour, day_of_week, month) -> dict:
        """Return historical aggregate features from the precomputed lookup."""
        if not self.hist_aggs:
            return {}
        global_stats = self.hist_aggs.get("global", {})
        # Derive quantile aliases from the global entry (excludes "mean"); e.g. ["p50", ..., "p95"]
        q_aliases = [k for k in global_stats if k != "mean"]
        result: dict = {}

        def lookup(section_key: str, row_key: str, prefix: str) -> None:
            row = self.hist_aggs.get(section_key, {}).get(row_key, {})
            result[f"{prefix}_n"] = float(row.get(f"{prefix}_n") or 0.0)
            result[f"{prefix}_mean"] = float(row.get(f"{prefix}_mean") or global_stats.get("mean", 0.0))
            for alias in q_aliases:
                result[f"{prefix}_{alias}"] = float(
                    row.get(f"{prefix}_{alias}") or global_stats.get(alias, 0.0)
                )

        s = str(bpuic) if bpuic is not None else None
        l = str(line_text) if line_text is not None else None
        h = str(int(hour)) if hour is not None else None
        d = str(int(day_of_week)) if day_of_week is not None else None
        m = str(int(month)) if month is not None else None

        if s:
            lookup("by_bpuic", s, "hist_stop")
        if l:
            lookup("by_line_text", l, "hist_line")
        if s and h:
            lookup("by_bpuic__hour", f"{s}__{h}", "hist_stop_hour")
        if l and h:
            lookup("by_line_text__hour", f"{l}__{h}", "hist_line_hour")
        if d and h:
            lookup("by_day_of_week__hour", f"{d}__{h}", "hist_dow_hour")
        if s and d:
            lookup("by_bpuic__day_of_week", f"{s}__{d}", "hist_stop_dow")
        if s and m:
            lookup("by_bpuic__month", f"{s}__{m}", "hist_stop_month")

        return result

    def _get_calendar_features(self, travel_date, bpuic=None) -> dict:
        """Return canton-specific calendar flags for a travel date.

        Looks up the stop's canton from stop_canton, then queries the holidays
        library for that canton. Falls back to ZH (conservative) when the canton
        is unknown.
        """
        empty = {c: 0 for c in CalendarDataHandler.CALENDAR_NUMERIC_COLS}
        if travel_date is None:
            return empty
        try:
            date_str = travel_date if isinstance(travel_date, str) else travel_date.isoformat()
            if date_str.lower() in _DAY_NAME_TO_DOW:
                return empty
            canton_code = self.stop_canton.get(str(bpuic)) if bpuic is not None else None
            return CalendarDataHandler.get_features_for_date(date_str, canton_code=canton_code)
        except Exception:
            return empty

    def _column_for_q(self, q: float) -> str:
        """Return the quantile column name for the smallest available level >= q."""
        q_norm = float(q) if float(q) <= 1.0 else float(q) / 100.0
        for level in sorted(self.quantile_columns):
            if q_norm <= level:
                return self.quantile_columns[level]
        return self.quantile_columns[max(self.quantile_columns)]

    @staticmethod
    def _norm_day(value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            mapped = _DAY_NAME_TO_DOW.get(value.strip().lower())
            if mapped is not None:
                return mapped
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _month_from_route(route: Mapping[str, object]) -> Optional[int]:
        value = route.get("travel_date")
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.month
        if isinstance(value, date):
            return value.month
        text = str(value).strip().lower()
        if text in _DAY_NAME_TO_DOW:
            return None
        try:
            return datetime.fromisoformat(text).month
        except ValueError:
            return None

    @staticmethod
    def _day_type_from_calendar(calendar_features: dict) -> int:
        """Map calendar flags to a single ordinal: 0=regular, 1=holiday, 2=near-holiday."""
        if calendar_features.get("is_public_holiday", 0):
            return 1
        if (calendar_features.get("is_bridge_day", 0)
                or calendar_features.get("is_day_before_public_holiday", 0)
                or calendar_features.get("is_day_after_public_holiday", 0)):
            return 2
        return 0

    @staticmethod
    def _secs_to_time(value: float) -> str:
        seconds = int(round(value))
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

    @staticmethod
    def _load_json(path: str) -> dict:
        candidate = Path(path)
        if not candidate.exists():
            return {}
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return {}
