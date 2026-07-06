from __future__ import annotations

"""XGBoost quantile regression delay-model trainer (distributed via SparkXGBRegressor).

Training runs fully distributed on the cluster using SparkXGBRegressor on a
stratified 10% sample of the data. After fitting each quantile model the underlying
XGBoost Booster and StringIndexer label vocabularies are extracted and saved
locally as JSON so inference needs no Spark at routing time.

See MODEL_IDEA.md for the reasoning behind global (non-region-scoped) training
and the choice of SparkXGBRegressor over driver-side xgb.train().
"""

from datetime import date as _date, datetime as _datetime
import json
from pathlib import Path
import time
from typing import Iterable

from src.config.settings import ProjectSettings, get_settings
from src.data.calendar_data import CalendarDataHandler
from src.data.delay_data_handler import DelayDataHandler
from src.data.weather_data import WeatherDataHandler
from .model_artifacts import ModelArtifacts


class DelayModelTrainer:
    """Train per-quantile XGBoost models for arrival-delay prediction."""

    LABEL_COL = "label_delay_min"
    CATEGORICAL_COLS = ["bpuic", "line_text", "hour", "day_of_week", "month", "day_type"]
    HIST_AGG_SPECS = [
        (["bpuic"],                "hist_stop"),
        (["line_text"],            "hist_line"),
        (["bpuic", "hour"],        "hist_stop_hour"),
        (["line_text", "hour"],    "hist_line_hour"),
        (["day_of_week", "hour"],  "hist_dow_hour"),
        (["bpuic", "day_of_week"], "hist_stop_dow"),
        (["bpuic", "month"],       "hist_stop_month"),
    ]
    HIST_QUANTILE_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

    def __init__(
        self,
        spark=None,
        settings: ProjectSettings | None = None,
        artifacts: ModelArtifacts | None = None,
        data_handler: DelayDataHandler | None = None,
    ):
        self.settings = settings or get_settings()
        self.artifacts = artifacts or ModelArtifacts(settings=self.settings)
        self.data_handler = data_handler or DelayDataHandler(spark=spark, settings=self.settings)

    def train(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        region_uuids: Iterable[str] | None = None,
        operator_ids: Iterable[str] | None = None,
        train_start_date: str | None = None,
        train_end_date: str | None = None,
        force_rebuild_data: bool = False,
        force_rebuild_region_stops: bool = False,
        force_retrain: bool = True,
        include_weather: bool = True,
        include_calendar: bool = True,
        sample_fraction: float | None = None,
        append_data_range: tuple[str, str] | None = None,
        region_bpuics: set[str] | None = None,
        region_artifacts: ModelArtifacts | None = None,
    ) -> dict[str, object]:
        """Train one XGBoost quantile model per configured quantile level.

        region_uuids is accepted for API compatibility but ignored: training always
        uses all available data stored under the 'global' artifact namespace.
        See MODEL_IDEA.md for rationale.
        """
        spark = self.data_handler.get_spark()
        artifacts = self.artifacts.global_delay_artifacts()
        artifacts.ensure_local_base_dir()

        feature_path = artifacts.delay_feature_data_path()
        train_start_date = train_start_date or self.settings.final_train_start_date
        train_end_date = train_end_date or self.settings.final_train_end_date
        quantile_levels = list(self.settings.xgb_quantile_levels)
        sample_fraction = sample_fraction if sample_fraction is not None else self.settings.sample_fraction

        rebuild_feature_data = (
            force_rebuild_data
            or force_rebuild_region_stops
            or not self._path_exists(spark, feature_path)
        )

        def _build_and_write(s_date, e_date, write_mode="overwrite", dynamic_overwrite=False):
            df, cat_cols, num_cols = self.data_handler.build_feature_input_data(
                start_date=s_date,
                end_date=e_date,
                region_uuids=None,
                operator_ids=operator_ids or self.settings.operator_ids,
                include_weather=include_weather,
                include_calendar=include_calendar,
                force_rebuild_region_stops=force_rebuild_region_stops,
                sample_fraction=sample_fraction,
                sample_seed=self.settings.sample_seed,
            )
            # dynamic_overwrite=True (incremental append): only replaces the partitions
            # being written, leaving other existing partitions intact.
            # dynamic_overwrite=False (full rebuild): static mode deletes the entire
            # output directory first, preventing stale files / schema contamination from
            # a previous write with a different schema.
            overwrite_mode = "dynamic" if dynamic_overwrite else "static"
            spark.conf.set("spark.sql.sources.partitionOverwriteMode", overwrite_mode)
            (
                df.withColumn("part_year", F.year("operating_day"))
                .withColumn("part_month", F.month("operating_day"))
                .write.mode(write_mode)
                .partitionBy("part_year", "part_month")
                .parquet(feature_path)
            )
            return cat_cols, num_cols

        if append_data_range is not None and self._path_exists(spark, feature_path):
            # Incremental: add a new date range to the existing partitioned parquet.
            app_start, app_end = append_data_range
            _build_and_write(app_start, app_end, write_mode="overwrite", dynamic_overwrite=True)
            feature_df = self._read_feature_df(spark, feature_path)
            categorical_cols = list(self.CATEGORICAL_COLS)
            numeric_cols = self._infer_numeric_cols(feature_df, categorical_cols)
        elif rebuild_feature_data:
            categorical_cols, numeric_cols = _build_and_write(
                start_date or train_start_date or self.settings.final_train_start_date,
                end_date or self.settings.model_end_date,
            )
            feature_df = self._read_feature_df(spark, feature_path)
        else:
            feature_df = self._read_feature_df(spark, feature_path)
            categorical_cols = list(self.CATEGORICAL_COLS)
            numeric_cols = self._infer_numeric_cols(feature_df, categorical_cols)

        metadata_path = Path(artifacts.model_metadata_path())
        all_local_exist = all(
            Path(artifacts.local_xgb_model_json_path(q)).exists() for q in quantile_levels
        )
        models_exist = not force_retrain and all_local_exist and metadata_path.exists()

        if models_exist:
            return {
                "feature_data_path": feature_path,
                "region_key": artifacts.region_key,
                "trained": False,
                "reason": "global model artifacts already exist",
            }

        train_raw_df = (
            feature_df
            .filter(F.col("operating_day").between(train_start_date, train_end_date))
            .persist(StorageLevel.DISK_ONLY)
        )
        train_rows = train_raw_df.count()
        if train_rows == 0:
            raise ValueError(f"No training rows between {train_start_date} and {train_end_date}.")

        start_hist = time.time()
        train_enriched_df, hist_numeric_cols = self.add_historical_delay_features(
            train_raw_df, train_raw_df, label_col=self.LABEL_COL,
        )
        hist_time = time.time() - start_hist

        all_numeric_cols = list(numeric_cols) + hist_numeric_cols
        train_enriched_df = self.prepare_ml_columns(
            train_enriched_df, categorical_cols, all_numeric_cols, self.LABEL_COL,
        ).persist(StorageLevel.DISK_ONLY)
        train_enriched_df.count()

        import xgboost as xgb
        import numpy as np

        train_times: dict[float, float] = {}
        category_labels: dict[str, list[str]] = {}
        loss_curves: dict[str, dict] = {}
        best_rounds: dict[str, int] = {}

        # Fit preprocessing stages once (vocabulary is quantile-independent).
        # Used to transform the 1% early-stopping sample without running full
        # pipeline.fit() first, which avoids the chicken-and-egg dependency.
        _fitted_prep = self.build_preprocessing_pipeline(
            categorical_cols, all_numeric_cols
        ).fit(train_enriched_df)

        # Sample 1% of training data for convergence diagnostics. The full
        # sample drives the throwaway booster — no eval split needed because
        # elbow detection on train loss is used instead of held-out early
        # stopping (a small held-out set produces too-noisy quantile loss
        # estimates to reliably trigger patience-based stopping).
        _sample_raw = train_enriched_df.sample(fraction=0.01, seed=42).persist(StorageLevel.DISK_ONLY)
        _sample_raw.count()

        def _to_dmatrix(df):
            for stage in _fitted_prep.stages:
                df = stage.transform(df)
            pd_df = df.select("features", self.LABEL_COL).toPandas()
            X = np.vstack([v.toArray() for v in pd_df["features"]])
            y = pd_df[self.LABEL_COL].values
            return xgb.DMatrix(X, label=y)

        _curve_dtrain = _to_dmatrix(_sample_raw)
        _sample_raw.unpersist(blocking=False)

        for q in quantile_levels:
            q_key = f"q{int(round(q * 100)):03d}"

            # Run the throwaway to completion on the 1% sample, then find the
            # elbow in the train loss curve: the round where per-round improvement
            # drops below 0.3% of total gain. Apply a 1.5x buffer (1% sample
            # converges slightly faster than the full dataset) and cap at
            # xgb_num_round.
            _es_losses: dict = {}
            xgb.train(
                {
                    "objective": "reg:quantileerror",
                    "quantile_alpha": float(q),
                    "max_depth": self.settings.xgb_max_depth,
                    "eta": self.settings.xgb_eta,
                    "subsample": self.settings.xgb_subsample,
                    "colsample_bytree": self.settings.xgb_colsample_bytree,
                    "min_child_weight": self.settings.xgb_min_child_weight,
                    "tree_method": "hist",
                    "seed": 42,
                },
                _curve_dtrain,
                num_boost_round=self.settings.xgb_num_round,
                evals=[(_curve_dtrain, "train")],
                evals_result=_es_losses,
                verbose_eval=False,
            )
            train_vals = list(next(iter(_es_losses.get("train", {}).values()), []))
            elbow = self._find_elbow(train_vals)
            best_round = min(int(elbow * 1.5) + 1, self.settings.xgb_num_round)
            best_rounds[q_key] = best_round
            loss_curves[q_key] = {
                "train": train_vals,
                "best_round": best_round,
            }
            print(f"  q={q}: elbow at round {elbow} → best_round={best_round} (max={self.settings.xgb_num_round})")

            # Train real distributed model using the per-quantile optimal round count
            pipeline = self.build_xgb_quantile_pipeline(
                q, categorical_cols, all_numeric_cols, num_round=best_round
            )
            t0 = time.time()
            fitted = pipeline.fit(train_enriched_df)
            train_times[q] = time.time() - t0

            # Save full PipelineModel to HDFS for reproducibility and offline evaluation
            fitted.write().overwrite().save(artifacts.xgb_model_path(q))

            # Extract XGBoost Booster and save as local JSON for fast driver-only inference
            booster = fitted.stages[-1].get_booster()
            local_path = Path(artifacts.local_xgb_model_json_path(q))
            local_path.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(local_path))

            # Capture StringIndexer vocabularies once (same across quantiles)
            if not category_labels:
                indexer_model = fitted.stages[0]
                category_labels = {
                    col: list(indexer_model.labelsArray[i])
                    for i, col in enumerate(categorical_cols)
                }

        self.save_hist_aggs_artifact(train_raw_df, artifacts)
        self.save_climatological_weather_artifact(
            train_raw_df.select("bpuic").distinct(), artifacts, spark,
        )

        # feature_cols order must exactly match the VectorAssembler input used during training
        index_cols = [f"{col}_index" for col in categorical_cols]
        feature_cols = index_cols + all_numeric_cols

        metadata = {
            "region_key": artifacts.region_key,
            "label_col": self.LABEL_COL,
            "categorical_cols": categorical_cols,
            "base_numeric_cols": list(numeric_cols),
            "hist_numeric_cols": hist_numeric_cols,
            "all_numeric_cols": all_numeric_cols,
            "feature_cols": feature_cols,
            "category_labels": category_labels,
            "weather_bus_cols": list(WeatherDataHandler.WEATHER_BUS_COLS) if include_weather else [],
            "calendar_numeric_cols": list(CalendarDataHandler.CALENDAR_NUMERIC_COLS) if include_calendar else [],
            "quantile_levels": quantile_levels,
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "model_type": "XGBoostQuantileRegressor",
            "xgb_num_round": self.settings.xgb_num_round,
            "xgb_max_depth": self.settings.xgb_max_depth,
            "xgb_eta": self.settings.xgb_eta,
            "best_rounds_per_quantile": best_rounds,
        }
        self.write_metadata(metadata, artifacts=artifacts)
        if region_bpuics is not None and region_artifacts is not None:
            self.save_precomputed_delays(
                train_enriched_df, region_artifacts, categorical_cols, all_numeric_cols,
                region_bpuics={str(b) for b in region_bpuics},
            )

        t_precomp = time.perf_counter()
        try:
            district_to_bpuics = self._fetch_district_bpuics()
            n = self._save_all_precomputed_delays_from_df(
                train_enriched_df, district_to_bpuics, categorical_cols, all_numeric_cols,
            )
            print(f"  precomputed delays: {len(n)} district files in {time.perf_counter() - t_precomp:.1f}s")
        except Exception as exc:
            print(f"  Warning: could not build district precomputed delays: {exc}")

        curves_path = self._save_train_loss_curves(
            loss_curves, artifacts,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            training_rows=int(train_rows),
        )

        train_enriched_df.unpersist(blocking=False)
        train_raw_df.unpersist(blocking=False)

        return {
            "feature_data_path": feature_path,
            "region_key": artifacts.region_key,
            "training_rows": int(train_rows),
            "hist_feature_time_sec": float(hist_time),
            "train_times_sec": {str(q): float(v) for q, v in train_times.items()},
            "train_loss_curves_path": curves_path,
            "trained": True,
        }

    def evaluate(
        self,
        train_start_date: str | None = None,
        train_end_date: str | None = None,
        val_periods: list[tuple[str, str]] | None = None,
    ) -> dict:
        """Compute pinball loss on held-out data using the trained Spark PipelineModels.

        Historical aggregate features are derived from training data only (no leakage).
        Results are saved to eval_results.json so they persist across sessions.

        Args:
            train_start_date: First date of training data used for hist-agg computation.
            train_end_date: Last date of training data.
            val_periods: List of (start, end) date pairs. Defaults to Oct 2025 – Jan 2026.
                Aug–Sep 2025 are absent from the source data and act as a natural buffer.
        """
        from pyspark.ml import PipelineModel

        spark = self.data_handler.get_spark()
        artifacts = self.artifacts.global_delay_artifacts()
        quantile_levels = list(self.settings.xgb_quantile_levels)

        train_start_date = train_start_date or self.settings.final_train_start_date
        train_end_date = train_end_date or self.settings.final_train_end_date

        if val_periods is None:
            val_periods = [("2025-10-01", "2026-01-31")]

        feature_path = artifacts.delay_feature_data_path()
        feature_df = self._read_feature_df(spark, feature_path)
        categorical_cols = list(self.CATEGORICAL_COLS)
        numeric_cols = self._infer_numeric_cols(feature_df, categorical_cols)

        train_raw_df = (
            feature_df
            .filter(F.col("operating_day").between(train_start_date, train_end_date))
            .persist(StorageLevel.DISK_ONLY)
        )
        train_raw_df.count()

        val_raw_df = None
        for start, end in val_periods:
            period = feature_df.filter(F.col("operating_day").between(start, end))
            val_raw_df = period if val_raw_df is None else val_raw_df.union(period)
        val_raw_df = val_raw_df.persist(StorageLevel.DISK_ONLY)
        val_rows = val_raw_df.count()
        if val_rows == 0:
            raise ValueError(f"No validation rows found for periods: {val_periods}")
        print(f"Validation rows: {val_rows:,}")

        val_enriched_df, hist_numeric_cols = self.add_historical_delay_features(
            train_raw_df, val_raw_df, label_col=self.LABEL_COL,
        )
        all_numeric_cols = list(numeric_cols) + hist_numeric_cols
        val_enriched_df = self.prepare_ml_columns(
            val_enriched_df, categorical_cols, all_numeric_cols, self.LABEL_COL,
        ).persist(StorageLevel.DISK_ONLY)
        val_enriched_df.count()

        pinball_losses: dict[str, float] = {}
        for q in quantile_levels:
            t0 = time.time()
            fitted = PipelineModel.load(artifacts.xgb_model_path(q))
            preds = fitted.transform(val_enriched_df)
            pinball = preds.select(
                F.avg(
                    F.when(
                        F.col(self.LABEL_COL) >= F.col("prediction"),
                        F.lit(q) * (F.col(self.LABEL_COL) - F.col("prediction")),
                    ).otherwise(
                        F.lit(1.0 - q) * (F.col("prediction") - F.col(self.LABEL_COL)),
                    )
                )
            ).first()[0]
            pinball_losses[str(q)] = round(float(pinball or 0.0), 4)
            print(f"  q={q}: pinball loss = {pinball_losses[str(q)]:.4f}  ({time.time() - t0:.0f}s)")

        train_raw_df.unpersist(blocking=False)
        val_raw_df.unpersist(blocking=False)
        val_enriched_df.unpersist(blocking=False)

        result = {
            "train_period": f"{train_start_date} → {train_end_date}",
            "val_periods": val_periods,
            "val_rows": val_rows,
            "val_pinball_loss": pinball_losses,
        }
        path = Path(artifacts.eval_results_json_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Eval results saved to {path}")
        return result

    def compare_baselines(
        self,
        train_start_date: str | None = None,
        train_end_date: str | None = None,
        val_periods: list[tuple[str, str]] | None = None,
        force_recompute: bool = False,
    ) -> dict:
        """Compare XGBoost quantile models against simpler baselines using pinball loss.

        Baselines:
          global_pct  – for each q, predict the training-set q-th percentile (a constant).
          group_hist  – predict per-(bpuic, line_text, hour) q-th percentile from training,
                        falling back to global for unseen groups. Approximates the approach
                        used in the main branch (historical-average lookup).

        Results are cached in baseline_results.json. Subsequent calls return the cached
        result immediately (no Spark jobs) unless force_recompute=True.
        """

        artifacts = self.artifacts.global_delay_artifacts()
        cache_path = Path(artifacts.baseline_results_json_path())
        if not force_recompute and cache_path.exists():
            print(f"Loading cached baseline results from {cache_path}")
            return json.loads(cache_path.read_text(encoding="utf-8"))

        spark = self.data_handler.get_spark()
        quantile_levels = list(self.settings.xgb_quantile_levels)

        train_start_date = train_start_date or self.settings.final_train_start_date
        train_end_date = train_end_date or self.settings.final_train_end_date
        if val_periods is None:
            val_periods = [("2025-10-01", "2026-01-31")]

        feature_path = artifacts.delay_feature_data_path()
        feature_df = self._read_feature_df(spark, feature_path)
        categorical_cols = list(self.CATEGORICAL_COLS)
        numeric_cols = self._infer_numeric_cols(feature_df, categorical_cols)

        train_raw_df = (
            feature_df
            .filter(F.col("operating_day").between(train_start_date, train_end_date))
            .persist(StorageLevel.DISK_ONLY)
        )
        train_raw_df.count()

        val_raw_df = None
        for start, end in val_periods:
            period = feature_df.filter(F.col("operating_day").between(start, end))
            val_raw_df = period if val_raw_df is None else val_raw_df.union(period)
        val_raw_df = val_raw_df.persist(StorageLevel.DISK_ONLY)
        val_rows = val_raw_df.count()
        if val_rows == 0:
            raise ValueError(f"No validation rows found for periods: {val_periods}")
        print(f"Validation rows: {val_rows:,}")

        label_col = self.LABEL_COL

        def _pinball_expr(pred_col, q: float):
            return F.avg(
                F.when(
                    F.col(label_col) >= pred_col,
                    F.lit(float(q)) * (F.col(label_col) - pred_col),
                ).otherwise(
                    F.lit(1.0 - float(q)) * (pred_col - F.col(label_col)),
                )
            )

        # ─── 1. Global percentile baseline ───────────────────────────────────
        print("\nBaseline 1/2: Global percentile")
        global_pct_row = train_raw_df.agg(
            *[F.expr(f"percentile_approx({label_col}, {float(q)})").alias(f"p{int(round(q*100)):03d}")
              for q in quantile_levels]
        ).first().asDict()

        global_pct_losses = {}
        global_loss_row = val_raw_df.agg(
            *[_pinball_expr(F.lit(float(global_pct_row[f"p{int(round(q*100)):03d}"] or 0.0)), q)
              .alias(f"loss_{int(round(q*100)):03d}")
              for q in quantile_levels]
        ).first().asDict()
        for q in quantile_levels:
            loss_key = f"loss_{int(round(q*100)):03d}"
            global_pct_losses[str(q)] = round(float(global_loss_row[loss_key] or 0.0), 4)
            print(f"  q={q}: {global_pct_losses[str(q)]:.4f}")

        # ─── 2. Per-group historical baseline (main-branch approximation) ─────
        # Groups by (bpuic, line_text, hour) — same granularity as the main branch's
        # line_stop_hour level in its 8-level fallback lookup.
        print("\nBaseline 2/2: Per-group historical percentile (bpuic, line_text, hour)")
        group_keys = ["bpuic", "line_text", "hour"]
        group_pct_df = (
            train_raw_df
            .groupBy(*group_keys)
            .agg(*[F.expr(f"percentile_approx({label_col}, {float(q)})").alias(f"grp_{int(round(q*100)):03d}")
                   for q in quantile_levels])
            .persist(StorageLevel.DISK_ONLY)
        )
        group_pct_df.count()

        val_with_grp = val_raw_df.join(group_pct_df, on=group_keys, how="left")
        group_hist_losses = {}
        grp_loss_row = val_with_grp.agg(
            *[_pinball_expr(
                F.coalesce(F.col(f"grp_{int(round(q*100)):03d}"),
                           F.lit(float(global_pct_row[f"p{int(round(q*100)):03d}"] or 0.0))),
                q,
            ).alias(f"loss_{int(round(q*100)):03d}")
              for q in quantile_levels]
        ).first().asDict()
        for q in quantile_levels:
            loss_key = f"loss_{int(round(q*100)):03d}"
            group_hist_losses[str(q)] = round(float(grp_loss_row[loss_key] or 0.0), 4)
            print(f"  q={q}: {group_hist_losses[str(q)]:.4f}")
        group_pct_df.unpersist(blocking=False)
        train_raw_df.unpersist(blocking=False)
        val_raw_df.unpersist(blocking=False)

        result = {
            "train_period": f"{train_start_date} → {train_end_date}",
            "val_periods": val_periods,
            "val_rows": val_rows,
            "baselines": {
                "global_pct": global_pct_losses,
                "group_hist": group_hist_losses,
            },
        }
        path = Path(artifacts.baseline_results_json_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nBaseline results saved to {path}")
        return result

    def model_exists(self) -> bool:
        """Check whether global delay model artifacts are ready for inference."""
        artifacts = self.artifacts.global_delay_artifacts()
        return (
            all(Path(artifacts.local_xgb_model_json_path(q)).exists() for q in self.settings.xgb_quantile_levels)
            and Path(artifacts.model_metadata_path()).exists()
        )

    def model_exists_for_region(self, region_uuids=None) -> bool:
        """Deprecated — models are always global. Use model_exists() instead."""
        return self.model_exists()

    def save_precomputed_delays(
        self,
        train_enriched_df,
        artifacts: ModelArtifacts,
        categorical_cols: list[str],
        all_numeric_cols: list[str],
        region_bpuics: set[str] | None = None,
    ) -> str:
        """Pre-compute XGBoost predictions for every training combo; save as a lookup JSON.

        When region_bpuics is provided the output is filtered to that stop set and the
        1M cap is dropped — a region has far fewer combinations than all of Switzerland.
        When region_bpuics is None the global cap is kept as a safety net.
        """
        from src.models.delay_model import DelayModel

        numeric_available = [c for c in all_numeric_cols if c in train_enriched_df.columns]

        filtered_df = train_enriched_df
        if region_bpuics is not None:
            filtered_df = filtered_df.filter(
                F.col("bpuic").cast("string").isin(list(region_bpuics))
            )

        precomp_df = (
            filtered_df
            .select(*categorical_cols, *numeric_available)
            .groupBy(*categorical_cols)
            .agg(*[F.avg(c).alias(c) for c in numeric_available])
            .fillna(0.0)
        )
        if region_bpuics is None:
            precomp_df = precomp_df.limit(1_000_000)

        spark = train_enriched_df.sparkSession
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        feature_rows = precomp_df.toPandas().to_dict("records")

        global_artifacts = ModelArtifacts(settings=self.settings).global_delay_artifacts()
        model = DelayModel.load(artifacts=global_artifacts)
        quantile_levels = sorted(model.xgb_models.keys())
        combo_keys = [
            "__".join(str(row.get(c, "")) for c in categorical_cols)
            for row in feature_rows
        ]

        all_preds = model._predict_xgb(feature_rows)

        lookup = {
            key: [round(all_preds[q][i], 2) for q in quantile_levels]
            for i, key in enumerate(combo_keys)
        }

        path = Path(artifacts.precomputed_delays_json_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lookup), encoding="utf-8")
        return str(path)

    def build_region_precomputed_delays(
        self,
        region_bpuics: set[str],
        region_artifacts: ModelArtifacts,
        train_start_date: str | None = None,
        train_end_date: str | None = None,
    ) -> str:
        """Build per-region precomputed delay lookup without retraining models.

        Called from prepare() when the region-specific file is missing — e.g. when
        someone clones the repo with pre-trained model JSONs but no precomputed file yet. Reads the full feature parquet (needed for correct global
        hist-feature computation), filters to region stops after enrichment, and saves
        to artifacts/precomputed_delays/{region_key}.json.
        """
        spark = self.data_handler.get_spark()
        global_artifacts = ModelArtifacts(settings=self.settings).global_delay_artifacts()
        feature_path = global_artifacts.delay_feature_data_path()

        start = train_start_date or self.settings.final_train_start_date
        end = train_end_date or self.settings.final_train_end_date

        feature_df = self._read_feature_df(spark, feature_path)
        categorical_cols = list(self.CATEGORICAL_COLS)
        numeric_cols = self._infer_numeric_cols(feature_df, categorical_cols)

        train_raw_df = (
            feature_df
            .filter(F.col("operating_day").between(start, end))
            .persist(StorageLevel.DISK_ONLY)
        )
        train_raw_df.count()

        train_enriched_df, hist_numeric_cols = self.add_historical_delay_features(
            train_raw_df, train_raw_df, label_col=self.LABEL_COL,
        )
        all_numeric_cols = list(numeric_cols) + hist_numeric_cols
        train_enriched_df = self.prepare_ml_columns(
            train_enriched_df, categorical_cols, all_numeric_cols, self.LABEL_COL,
        ).persist(StorageLevel.DISK_ONLY)
        train_enriched_df.count()

        path = self.save_precomputed_delays(
            train_enriched_df, region_artifacts, categorical_cols, all_numeric_cols,
            region_bpuics={str(b) for b in region_bpuics},
        )

        train_enriched_df.unpersist(blocking=False)
        train_raw_df.unpersist(blocking=False)
        return path

    def _fetch_district_bpuics(self) -> dict[str, set[str]]:
        """Query Trino for all CH districts and the stop bpuics they contain."""
        import pandas as pd
        from src.data.csa_data_handler import CSADataHandler

        handler = CSADataHandler(settings=self.settings)
        query = f"""
        SELECT DISTINCT
            CAST(g.uuid AS VARCHAR) AS district_uuid,
            TRY_CAST(split_part(s.stop_id, ':', 1) AS INTEGER) AS bpuic
        FROM {handler._table('src_geo')} g
        JOIN {handler._table('src_stops')} s
            ON ST_Contains(ST_GeomFromBinary(g.wkb_geometry), ST_Point(s.stop_lon, s.stop_lat))
        WHERE g.country = 'CH'
            AND g.level = 'district'
            AND s.pub_date = DATE '{handler.max_pub_date}'
            AND split_part(s.stop_id, ':', 1) LIKE '85%'
            AND TRY_CAST(split_part(s.stop_id, ':', 1) AS INTEGER) IS NOT NULL
        """
        df = pd.read_sql(query, handler.conn)
        district_to_bpuics: dict[str, set[str]] = {}
        for _, row in df.iterrows():
            district_to_bpuics.setdefault(str(row["district_uuid"]), set()).add(
                str(int(row["bpuic"]))
            )
        return district_to_bpuics

    def _save_all_precomputed_delays_from_df(
        self,
        train_enriched_df,
        district_to_bpuics: dict[str, set[str]],
        categorical_cols: list[str],
        all_numeric_cols: list[str],
        read_batch_size: int = 5,
    ) -> dict[str, str]:
        """Generate one precomputed lookup JSON per district from an already-loaded df.

        Does ONE Spark pass (join + group + write to temp Parquet), then reads
        the already-grouped (small) result back in batches of `read_batch_size`
        districts to stay under spark.driver.maxResultSize.
        """
        from src.models.delay_model import DelayModel

        numeric_available = [c for c in all_numeric_cols if c in train_enriched_df.columns]
        spark = train_enriched_df.sparkSession

        global_artifacts = ModelArtifacts(settings=self.settings).global_delay_artifacts()
        model = DelayModel.load(artifacts=global_artifacts)
        quantile_levels = sorted(model.xgb_models.keys())

        # Build full mapping once and do ONE join+group+write Spark job
        all_bpuic_rows = [
            (bpuic, district_uuid)
            for district_uuid, bpuics in district_to_bpuics.items()
            for bpuic in bpuics
        ]
        district_bpuic_df = spark.createDataFrame(all_bpuic_rows, ["bpuic_str", "district_uuid"])

        temp_path = self.settings.artifacts_base_path.rstrip("/") + "/tmp/precomp_districts"
        (
            train_enriched_df
            .withColumn("bpuic_str", F.col("bpuic").cast("string"))
            .join(district_bpuic_df, on="bpuic_str", how="inner")
            .select("district_uuid", *categorical_cols, *numeric_available)
            .groupBy("district_uuid", *categorical_cols)
            .agg(*[F.avg(c).alias(c) for c in numeric_available])
            .fillna(0.0)
            .write.partitionBy("district_uuid").parquet(temp_path, mode="overwrite")
        )
        print(f"  grouped data written to {temp_path}")

        # Read back the small grouped Parquet in batches — each read is cheap
        grouped_df = spark.read.parquet(temp_path)
        saved_paths: dict[str, str] = {}
        district_uuids = list(district_to_bpuics.keys())
        n_batches = (len(district_uuids) + read_batch_size - 1) // read_batch_size

        for batch_start in range(0, len(district_uuids), read_batch_size):
            batch_uuids = district_uuids[batch_start:batch_start + read_batch_size]
            batch_pandas = (
                grouped_df
                .filter(F.col("district_uuid").isin(batch_uuids))
                .toPandas()
            )

            for district_uuid in batch_uuids:
                group = batch_pandas[batch_pandas["district_uuid"] == str(district_uuid)]
                if group.empty:
                    continue
                rows = group.drop(columns=["district_uuid"]).to_dict("records")
                all_preds = model._predict_xgb(rows)
                combo_keys = [
                    "__".join(str(row.get(c, "")) for c in categorical_cols)
                    for row in rows
                ]
                lookup = {
                    key: [round(all_preds[q][i], 2) for q in quantile_levels]
                    for i, key in enumerate(combo_keys)
                }
                region_artifacts = ModelArtifacts(settings=self.settings).for_region([district_uuid])
                path = Path(region_artifacts.precomputed_delays_json_path())
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(lookup), encoding="utf-8")
                saved_paths[district_uuid] = str(path)

            print(
                f"  precomputed_delays batch {batch_start // read_batch_size + 1}/{n_batches}"
                f"  saved={len(saved_paths)}"
            )

        return saved_paths

    def build_all_region_precomputed_delays(
        self,
        train_start_date: str | None = None,
        train_end_date: str | None = None,
        force: bool = False,
    ) -> dict[str, str]:
        """Build precomputed delay lookups for all CH districts in one Spark pass.

        Queries Trino for all 134 CH district UUIDs, maps each to its contained
        stops, loads the enriched training data once in Spark, and saves one JSON
        file per district under artifacts/precomputed_delays/{region_key}.json.
        Skips districts whose file already exists unless force=True.
        """
        district_to_bpuics = self._fetch_district_bpuics()
        if not district_to_bpuics:
            return {}

        if not force:
            district_to_bpuics = {
                uuid: bpuics
                for uuid, bpuics in district_to_bpuics.items()
                if not Path(
                    ModelArtifacts(settings=self.settings)
                    .for_region([uuid])
                    .precomputed_delays_json_path()
                ).exists()
            }
        if not district_to_bpuics:
            print("  All district precomputed delay files already exist, skipping.")
            return {}

        print(f"  Building precomputed delays for {len(district_to_bpuics)} districts...")

        spark = self.data_handler.get_spark()
        global_artifacts = ModelArtifacts(settings=self.settings).global_delay_artifacts()
        feature_path = global_artifacts.delay_feature_data_path()
        start = train_start_date or self.settings.final_train_start_date
        end = train_end_date or self.settings.final_train_end_date

        feature_df = self._read_feature_df(spark, feature_path)
        categorical_cols = list(self.CATEGORICAL_COLS)
        numeric_cols = self._infer_numeric_cols(feature_df, categorical_cols)

        train_raw_df = (
            feature_df
            .filter(F.col("operating_day").between(start, end))
            .persist(StorageLevel.DISK_ONLY)
        )
        train_raw_df.count()

        train_enriched_df, hist_numeric_cols = self.add_historical_delay_features(
            train_raw_df, train_raw_df, label_col=self.LABEL_COL,
        )
        all_numeric_cols = list(numeric_cols) + hist_numeric_cols
        train_enriched_df = self.prepare_ml_columns(
            train_enriched_df, categorical_cols, all_numeric_cols, self.LABEL_COL,
        ).persist(StorageLevel.DISK_ONLY)
        train_enriched_df.count()

        saved_paths = self._save_all_precomputed_delays_from_df(
            train_enriched_df, district_to_bpuics, categorical_cols, all_numeric_cols,
        )

        train_enriched_df.unpersist(blocking=False)
        train_raw_df.unpersist(blocking=False)
        return saved_paths

    def build_preprocessing_pipeline(self, categorical_cols, numeric_cols):
        """StringIndexer + VectorAssembler only — no XGBoost stage.

        Used to transform the early-stopping sample before the quantile loop so
        DMatrices can be built without a full pipeline.fit() per quantile.
        """
        index_cols = [f"{col}_index" for col in categorical_cols]
        indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_cols, handleInvalid="keep")
        assembler = VectorAssembler(inputCols=index_cols + numeric_cols, outputCol="features", handleInvalid="keep")
        return Pipeline(stages=[indexer, assembler])

    def build_xgb_quantile_pipeline(self, quantile, categorical_cols, numeric_cols, num_round: int | None = None):
        """Pipeline: StringIndexer → VectorAssembler → SparkXGBRegressor.

        No OHE: XGBoost trees split efficiently on label-encoded integers, and OHE
        on high-cardinality columns like bpuic would blow up the feature space.
        StringIndexer vocabularies are saved to metadata so driver-side inference can
        replicate the same label encoding without Spark.

        num_round overrides xgb_num_round from settings when early stopping has
        determined the optimal round count for this quantile on the 1% sample.
        """
        rounds = num_round if num_round is not None else self.settings.xgb_num_round
        index_cols = [f"{col}_index" for col in categorical_cols]
        indexer = StringIndexer(
            inputCols=categorical_cols,
            outputCols=index_cols,
            handleInvalid="keep",
        )
        assembler = VectorAssembler(
            inputCols=index_cols + numeric_cols,
            outputCol="features",
            handleInvalid="keep",
        )
        regressor = SparkXGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=float(quantile),
            n_estimators=rounds,
            max_depth=self.settings.xgb_max_depth,
            learning_rate=self.settings.xgb_eta,
            subsample=self.settings.xgb_subsample,
            colsample_bytree=self.settings.xgb_colsample_bytree,
            min_child_weight=self.settings.xgb_min_child_weight,
            num_workers=self.settings.xgb_num_workers,
            features_col="features",
            label_col=self.LABEL_COL,
            prediction_col="prediction",
            random_state=42,
        )
        return Pipeline(stages=[indexer, assembler, regressor])

    @staticmethod
    def build_pipeline(model, categorical_cols, numeric_cols, regression_type="tree"):
        """Generic pipeline with OHE; kept for use with LinearRegression baselines."""
        index_cols = [f"{col}_index" for col in categorical_cols]
        encoded_cols = [f"{col}_vec" for col in categorical_cols]
        indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_cols, handleInvalid="keep")
        encoder = OneHotEncoder(
            inputCols=index_cols,
            outputCols=encoded_cols,
            handleInvalid="keep",
            dropLast=(regression_type == "linear"),
        )
        assembler = VectorAssembler(
            inputCols=encoded_cols + numeric_cols,
            outputCol="features",
            handleInvalid="keep",
        )
        return Pipeline(stages=[indexer, encoder, assembler, model])

    @classmethod
    def add_historical_delay_features(cls, train_df, target_df, label_col: str = LABEL_COL):
        """Build fold-safe historical delay aggregates from train_df only."""
        q_levels = cls.HIST_QUANTILE_LEVELS
        global_stats = train_df.agg(
            F.avg(label_col).alias("global_mean"),
            *[
                F.expr(f"percentile_approx({label_col}, {q})").alias(f"global_p{int(round(q * 100))}")
                for q in q_levels
            ],
        ).collect()[0]

        global_fallbacks = {"mean": float(global_stats["global_mean"] or 0.0)}
        for q in q_levels:
            key = f"p{int(round(q * 100))}"
            global_fallbacks[key] = float(global_stats[f"global_{key}"] or 0.0)

        out_df = target_df
        hist_cols = []
        for keys, prefix in cls.HIST_AGG_SPECS:
            agg_df = train_df.groupBy(*keys).agg(
                F.count("*").cast("double").alias(f"{prefix}_n"),
                F.avg(label_col).alias(f"{prefix}_mean"),
                *[
                    F.expr(f"percentile_approx({label_col}, {q})").alias(f"{prefix}_p{int(round(q * 100))}")
                    for q in q_levels
                ],
            )
            out_df = out_df.join(agg_df, on=keys, how="left")
            group_cols = [f"{prefix}_n", f"{prefix}_mean"] + [
                f"{prefix}_p{int(round(q * 100))}" for q in q_levels
            ]
            hist_cols.extend(group_cols)

        fill_values = {}
        for col in hist_cols:
            if col.endswith("_n"):
                fill_values[col] = 0.0
            else:
                for stat_suffix, val in global_fallbacks.items():
                    if col.endswith(f"_{stat_suffix}"):
                        fill_values[col] = val
                        break
        return out_df.fillna(fill_values), hist_cols

    @staticmethod
    def prepare_ml_columns(df, categorical_cols, numeric_cols, label_col):
        out_df = df
        for col in categorical_cols:
            # Replace both null and empty string — OHE rejects "" as a category label.
            out_df = out_df.withColumn(
                col,
                F.when(
                    F.col(col).isNull() | (F.col(col).cast("string") == F.lit("")),
                    F.lit("UNKNOWN"),
                ).otherwise(F.col(col).cast("string")),
            )
        for col in numeric_cols:
            out_df = out_df.withColumn(col, F.coalesce(F.col(col).cast("double"), F.lit(0.0)))
        return out_df.withColumn(label_col, F.col(label_col).cast("double"))

    def evaluate_predictions(self, pred_df):
        """Compute regression metrics. Call manually after fitting; not invoked during train()."""
        rmse = RegressionEvaluator(labelCol=self.LABEL_COL, predictionCol="prediction", metricName="rmse").evaluate(pred_df)
        mae = RegressionEvaluator(labelCol=self.LABEL_COL, predictionCol="prediction", metricName="mae").evaluate(pred_df)
        r2 = RegressionEvaluator(labelCol=self.LABEL_COL, predictionCol="prediction", metricName="r2").evaluate(pred_df)
        extra = (
            pred_df
            .withColumn("abs_error", F.abs(F.col("prediction") - F.col(self.LABEL_COL)))
            .withColumn("error", F.col("prediction") - F.col(self.LABEL_COL))
            .agg(
                F.expr("percentile_approx(abs_error, 0.90)").alias("p90_abs_error"),
                F.expr("percentile_approx(abs_error, 0.95)").alias("p95_abs_error"),
                F.avg("error").alias("bias"),
                F.avg(F.when(F.col("prediction") < F.col(self.LABEL_COL), 1.0).otherwise(0.0)).alias("underprediction_rate"),
            )
            .collect()[0]
        )
        return {
            "rmse": float(rmse), "mae": float(mae), "r2": float(r2),
            "p90_abs_error": float(extra["p90_abs_error"]),
            "p95_abs_error": float(extra["p95_abs_error"]),
            "bias": float(extra["bias"]),
            "underprediction_rate": float(extra["underprediction_rate"]),
        }

    def save_hist_aggs_artifact(self, train_df, artifacts: ModelArtifacts) -> str:
        """Save per-group historical delay aggregates as a local JSON lookup."""
        q_levels = self.HIST_QUANTILE_LEVELS
        q_aliases = [f"p{int(round(q * 100))}" for q in q_levels]

        global_row = train_df.agg(
            F.avg(self.LABEL_COL).alias("mean"),
            *[F.expr(f"percentile_approx({self.LABEL_COL}, {q})").alias(alias)
              for q, alias in zip(q_levels, q_aliases)],
        ).collect()[0]

        global_entry: dict = {"mean": float(global_row["mean"] or 0.0)}
        for alias in q_aliases:
            global_entry[alias] = float(global_row[alias] or 0.0)

        result: dict = {"global": global_entry}

        for keys, prefix in self.HIST_AGG_SPECS:
            agg_df = train_df.groupBy(*keys).agg(
                F.count("*").cast("double").alias(f"{prefix}_n"),
                F.avg(self.LABEL_COL).alias(f"{prefix}_mean"),
                *[F.expr(f"percentile_approx({self.LABEL_COL}, {q})").alias(f"{prefix}_{alias}")
                  for q, alias in zip(q_levels, q_aliases)],
            )
            section_key = "by_" + "__".join(keys)
            rows: dict = {}
            for row in agg_df.collect():
                rd = row.asDict()
                composite = "__".join(str(rd[k]) for k in keys)
                entry = {
                    f"{prefix}_n": float(rd.get(f"{prefix}_n") or 0.0),
                    f"{prefix}_mean": float(rd.get(f"{prefix}_mean") or 0.0),
                }
                for alias in q_aliases:
                    entry[f"{prefix}_{alias}"] = float(rd.get(f"{prefix}_{alias}") or 0.0)
                rows[composite] = entry
            result[section_key] = rows

        path = Path(artifacts.hist_aggs_json_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")
        return str(path)

    def save_climatological_weather_artifact(
        self, bpuic_df, artifacts: ModelArtifacts, spark
    ) -> str | None:
        """Save (station_id, month) climatological averages, stop→station mapping, and stop→canton mapping."""
        try:
            weather_handler = WeatherDataHandler(spark=spark, settings=self.settings)
            weather_clean = weather_handler.clean_weather_history()
        except Exception:
            return None

        try:
            # Use all Swiss stops from the timetable, not just training stops, so that
            # bus stops absent from training data still get a nearest-station mapping
            # at inference time (avoids all-zero weather features for unseen stops).
            all_stops_bpuic_df = (
                spark.table(self.settings.timetable_stops_table)
                .select(F.split(F.col("stop_id"), ":").getItem(0).cast("string").alias("bpuic"))
                .distinct()
            )
            stop_station_df = weather_handler.build_stop_station_mapping(all_stops_bpuic_df)
        except Exception:
            return None

        clim_df = (
            weather_clean
            .withColumn("month", F.month("weather_hour"))
            .groupBy("station_id", "month")
            .agg(*[F.avg(col).alias(col) for col in WeatherDataHandler.WEATHER_NUMERIC_COLS])
        )

        clim_data: dict = {}
        for row in clim_df.collect():
            rd = row.asDict()
            key = f"{rd['station_id']}__{rd['month']}"
            clim_data[key] = {
                col: float(rd[col]) if rd.get(col) is not None else 0.0
                for col in WeatherDataHandler.WEATHER_NUMERIC_COLS
            }

        stop_station_data: dict = {}
        for row in stop_station_df.collect():
            rd = row.asDict()
            stop_station_data[str(rd["bpuic"])] = str(rd["nearest_station_id"])

        clim_path = Path(artifacts.climatological_weather_json_path())
        clim_path.parent.mkdir(parents=True, exist_ok=True)
        clim_path.write_text(json.dumps(clim_data), encoding="utf-8")

        ss_path = Path(artifacts.stop_station_mapping_json_path())
        ss_path.write_text(json.dumps(stop_station_data), encoding="utf-8")

        try:
            stop_canton_df = CalendarDataHandler(
                spark=spark, settings=self.settings
            ).build_stop_canton_mapping()
            stop_canton_data: dict = {
                str(row["bpuic"]): str(row["canton_code"])
                for row in stop_canton_df.collect()
            }
            sc_path = Path(artifacts.stop_canton_mapping_json_path())
            sc_path.write_text(json.dumps(stop_canton_data), encoding="utf-8")
        except Exception:
            pass

        return str(clim_path)

    def write_metadata(self, metadata: dict, artifacts: ModelArtifacts | None = None) -> str:
        artifacts = artifacts or self.artifacts
        path = Path(artifacts.model_metadata_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return str(path)

    @staticmethod
    def _find_elbow(values: list[float], window: int = 5) -> int:
        """Return round index where per-round improvement drops below 0.3% of total gain."""
        import numpy as _np
        total = values[0] - values[-1]
        if total <= 0:
            return len(values) - 1
        smoothed = _np.convolve(values, _np.ones(window) / window, mode="valid")
        deltas = _np.abs(_np.diff(smoothed))
        threshold = 0.003 * total
        for i, d in enumerate(deltas):
            if d < threshold:
                return i + window // 2
        return len(values) - 1

    def _save_train_loss_curves(
        self,
        loss_curves: dict[str, dict],
        artifacts: ModelArtifacts,
        train_start_date: str,
        train_end_date: str,
        training_rows: int,
    ) -> str:
        """Save per-round XGBoost loss curves to a dated JSON file.

        Each quantile entry contains {"train": [...], "best_round": int} where
        best_round is the elbow-detected optimal round (with 1.5x buffer), used
        as num_round for the real SparkXGBRegressor training.
        """
        today = _datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        payload = {
            "train_date": today,
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "training_rows": training_rows,
            "curves": loss_curves,
        }
        path = Path(artifacts.train_loss_curves_json_path(today))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def _read_feature_df(spark, feature_path: str):
        """Read the partitioned feature parquet, dropping the partition helper columns."""
        return spark.read.parquet(feature_path).drop("part_year", "part_month")

    @staticmethod
    def _infer_numeric_cols(df, categorical_cols: list[str]) -> list[str]:
        excluded = set(categorical_cols) | {"operating_day", "label_delay_min", "part_year", "part_month"}
        return [col for col in df.columns if col not in excluded]

    @staticmethod
    def _path_exists(spark, path: str) -> bool:
        try:
            jvm = spark._jvm
            hadoop_conf = spark._jsc.hadoopConfiguration()
            fs = jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
            return bool(fs.exists(jvm.org.apache.hadoop.fs.Path(path)))
        except Exception:
            return False


try:
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
    from pyspark.sql import functions as F
    from pyspark.storagelevel import StorageLevel
except ImportError:
    Pipeline = None
    RegressionEvaluator = None
    OneHotEncoder = None
    StringIndexer = None
    VectorAssembler = None
    F = None
    StorageLevel = None

try:
    from xgboost.spark import SparkXGBRegressor
except ImportError:
    SparkXGBRegressor = None
