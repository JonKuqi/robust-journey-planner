from __future__ import annotations

"""Model Updator — daily istdaten append and weekly model retraining.

Intended usage:
    updator = ModelUpdator(spark=spark)
    updator.init()   # one-time: copy shared istdaten, write initial date files
    updator.run()    # daily: append new data, retrain if a week has passed
"""

from contextlib import closing
from datetime import date, timedelta
from typing import Iterable

from src.config.settings import ProjectSettings, get_settings
from src.config.trino_connection import create_trino_connection
from src.data.istdaten_fetcher import IstDatenFetcher
from src.models.delay_model_trainer import DelayModelTrainer
from src.models.model_artifacts import ModelArtifacts
from src.models.model_artifacts_2 import (
    get_active_model,
    inactive_model,
    init_model_dirs,
    read_dates,
    update_active_model,
    write_dates,
    read_failed_days,
    write_failed_days,
    read_divergence_state,
    write_divergence_state
)


class ModelUpdator:

    RETRAIN_INTERVAL_DAYS = 7
    FETCH_RETRIES = 3                  # immediate retries per day before flagging
    FETCH_RETRY_DELAY_SECONDS = 30     # wait between immediate retries
    # Divergence: if the active model's MAE on recent data exceeds the baseline MAE
    # saved at training time by more than this fraction, trigger an early retrain.
    # e.g. 0.20 = MAE has grown by more than 20 % relative to baseline.
    DIVERGENCE_THRESHOLD = 0.20
    # How many of the most recent days to pull for the divergence evaluation.
    DIVERGENCE_LOOKBACK_DAYS = 3

    def __init__(
        self,
        spark=None,
        settings: ProjectSettings | None = None,
        conn=None,
        retrain_interval_days: int = RETRAIN_INTERVAL_DAYS,
        fetch_retries: int = FETCH_RETRIES,
        fetch_retry_delay: int = FETCH_RETRY_DELAY_SECONDS,
        divergence_threshold: float = DIVERGENCE_THRESHOLD,
        divergence_lookback_days: int = DIVERGENCE_LOOKBACK_DAYS,
    ):
        self.settings = settings or get_settings()
        self.spark = spark
        self.conn = conn or create_trino_connection(self.settings)
        self.retrain_interval_days = retrain_interval_days
        self.fetch_retries = fetch_retries
        self.fetch_retry_delay = fetch_retry_delay
        self.divergence_threshold = divergence_threshold
        self.divergence_lookback_days = divergence_lookback_days
        self.fetcher = IstDatenFetcher(settings=self.settings, conn=self.conn)
        init_model_dirs(self.settings)

    def init(self, force_copy: bool = False) -> None:
        """One-time setup: copy shared istdaten into user schema and write date tracking files.

        Call this once before the first run(). Safe to call again — skips the
        copy if the table already exists unless force_copy=True.
        """
        print("=== ModelUpdator.init() ===")

        spark = self._get_spark()
        self.fetcher.copy_from_shared_spark(spark=spark, force=force_copy)

        # Write istdaten_dates based on what's now in the user's table
        row = spark.sql(
            f"SELECT MIN(operating_day), MAX(operating_day)"
            f" FROM {self.settings.spark_user_istdaten_table}"
        ).first()
        start_date = row[0].isoformat() if row[0] else "2025-01-01"
        end_date   = row[1].isoformat() if row[1] else str(date.today())

        write_dates("istdaten_dates", start=start_date, end=end_date, settings=self.settings)
        print(f"  istdaten_dates written: START={start_date}  END={end_date}")

        write_failed_days("failed_istdaten_days", [], self.settings)

    def run(
        self,
        region_uuids: Iterable[str] | None = None,
        force_retrain: bool = False,
    ) -> dict:
        """Daily entry point.

        1. Retry any days that previously failed
        2. Fetch new istdaten days from the internet and append to {user_schema}.istdaten.
        3. Check if a week has passed since the end of training data, or if current model significantly deviates from a freshly built one.
        4. If true, train into the inactive model slot and flip the active pointer.
        """
        print("=== ModelUpdator.run() ===")

        retried = self._retry_failed_days()
        if retried:
            print(f"  Recovered {len(retried)} previously-failed day(s): {retried}")
        
        new_days, newly_failed = self._update_istdaten()
        print(f"  Appended {len(new_days)} new day(s).")
        if newly_failed:
            print(f"  WARNING: {len(newly_failed)} day(s) failed all retries and flagged: {newly_failed}")

        retrain_reason = None
        if force_retrain:
            retrain_reason = "force_retrain=True"
        elif self._should_retrain():
            retrain_reason = f"≥{self.retrain_interval_days} days since last training"
 
        if retrain_reason is None:
            print("  Skipping time-based retrain — running divergence check instead.")
            divergenceed, divergence_info = self._check_divergence()
            if divergenceed:
                retrain_reason = f"model divergence detected: {divergence_info}"
 
        if retrain_reason is None:
            print("  No retrain needed.")
            return {
                "retrained": False,
                "new_days": new_days,
                "retried_days": retried,
                "failed_days": newly_failed,
            }
 
        print(f"  Retrain triggered — reason: {retrain_reason}")
        
        slot = inactive_model(self.settings)
        print(f"  Retraining into slot: {slot}")

        artifacts = ModelArtifacts(
            base_path=f"{self.settings.local_artifacts_path}/model/{slot}",
            settings=self.settings,
        )

        istdaten_dates = read_dates("istdaten_dates", self.settings)
        train_start = istdaten_dates.get("START", self.settings.final_train_start_date)
        train_end   = istdaten_dates.get("END",   self.settings.final_train_end_date)

        training_settings = get_settings(istdaten_table=self.settings.spark_user_istdaten_table)
        trainer = DelayModelTrainer(
            spark=self._get_spark(),
            settings=training_settings,
            artifacts=artifacts,
        )
        train_result = trainer.train(
            train_start_date=train_start,
            train_end_date=train_end,
            region_uuids=region_uuids,
            force_retrain=True,
        )

        baseline_pinball = None
        eval_period_end   = train_end
        eval_period_start = (
            date.fromisoformat(train_end) - timedelta(days=self.divergence_lookback_days)
        ).isoformat()
        try:
            eval_result = trainer.evaluate(
                train_start_date=train_start,
                train_end_date=train_end,
                val_periods=[(eval_period_start, eval_period_end)],
            )
            # q=0.5 pinball loss is the closest analogue to MAE for quantile models
            # and is always present because 0.5 is in HIST_QUANTILE_LEVELS.
            baseline_pinball = eval_result.get("val_pinball_loss", {}).get("0.5")
            print(f"  Baseline pinball (q=0.5): {baseline_pinball}")
        except Exception as exc:
            print(f"  WARNING: post-train evaluation failed ({exc}) — baseline not saved.")

        if new_days:
            self._remove_oldest_days(len(new_days))
        
        update_active_model(slot, self.settings)

        self._write_divergence_state(
            active_slot=slot,
            baseline_pinball=baseline_pinball,
            train_end=train_end,
        )
        
        write_dates("model_trained_dates", start=train_start, end=train_end, settings=self.settings)
        print(f"  Active model switched to: {slot}")

        return {
            "retrained": True,
            "retrain_reason": retrain_reason,
            "slot": slot,
            "new_days": new_days,
            "retried_days": retried,
            "failed_days": newly_failed,
            "train_result": train_result,
            "baseline_pinball": baseline_pinball,
        }

    def _update_istdaten(self) -> tuple[list[str], list[str]]:
        """Fetch days not yet in the user's istdaten table and append them.

        Reads the last date from the Trino table, fetches every day from
        last+1 to yesterday that is available on the website, appends via
        Spark, and updates istdaten_dates.
        """
        table = self.settings.spark_user_istdaten_table

        row = self._get_spark().sql(f"SELECT MAX(operating_day) FROM {table}").first()
        last_date = row[0] if row and row[0] else None

        if last_date is None:
            print("  istdaten table empty — run init() first.")
            return [], []

        if isinstance(last_date, str):
            last_date = date.fromisoformat(last_date)

        yesterday = date.today() - timedelta(days=1)
        if last_date >= yesterday:
            return [], []

        available = self.fetcher._build_resource_index()
        appended: list[str] = []
        failed: list[str] = []

        current = last_date + timedelta(days=1)
        while current <= yesterday:
            day_str = current.isoformat()
            if day_str not in available:
                print(f"  {day_str} not on website yet, stopping.")
                break
            success = self._fetch_and_append_with_retry(day_str, table)
            if success:
                appended.append(day_str)
            else:
                failed.append(day_str)
            # try:
            #     rows = self.fetcher.fetch_day(day_str)
            #     self._append_rows_via_spark(rows, table)
            #     appended.append(day_str)
            #     print(f"  Appended {len(rows):,} rows for {day_str}.")
            # except Exception as exc:
            #     print(f"  Failed {day_str}: {exc}")
            current += timedelta(days=1)

        if appended:
            dates_file = read_dates("istdaten_dates", self.settings)
            write_dates(
                "istdaten_dates",
                start=dates_file.get("START", str(last_date)),
                end=appended[-1],
                settings=self.settings,
            )

        if failed:
            existing = read_failed_days("failed_istdaten_days", self.settings)
            write_failed_days("failed_istdaten_days", list(set(existing + failed)), self.settings)
        
        return appended, failed

    def _fetch_and_append_with_retry(self, day_str: str, table: str) -> bool:
        """Try to fetch and append a single day up to self.fetch_retries times.

        Returns True if the day was successfully written, False otherwise.
        """
        for attempt in range(1, self.fetch_retries + 1):
            try:
                rows = self.fetcher.fetch_day(day_str)
                self._append_rows_via_spark(rows, table)
                print(f"  Appended {len(rows):,} rows for {day_str} (attempt {attempt}).")
                return True
            except Exception as exc:
                print(f"  Attempt {attempt}/{self.fetch_retries} failed for {day_str}: {exc}")
                if attempt < self.fetch_retries:
                    time.sleep(self.fetch_retry_delay)
        print(f"  Giving up on {day_str} — will retry on next run.")
        return False

    def _remove_oldest_days(self, n: int) -> None:
        """Delete the n oldest operating_days from the user's istdaten table.
 
        Called after a retrain to keep the table at a stable size — one day
        removed for every new day that was appended this cycle.
        """
        if n <= 0:
            return
        table = self.settings.spark_user_istdaten_table
        try:
            spark = self._get_spark()
            rows = spark.sql(
                f"SELECT DISTINCT operating_day FROM {table}"
                f" ORDER BY operating_day ASC LIMIT {n}"
            ).collect()
            days_to_drop = [
                r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]) for r in rows
            ]

            if not days_to_drop:
                return

            dates_file = read_dates("istdaten_dates", self.settings)

            for day in days_to_drop:
                spark.sql(f"DELETE FROM {table} WHERE operating_day = DATE '{day}'")

            row = spark.sql(f"SELECT MIN(operating_day) FROM {table}").first()
            new_start = row[0].isoformat() if row and row[0] else dates_file.get("START")
            write_dates(
                "istdaten_dates",
                start=new_start,
                end=dates_file.get("END", str(date.today())),
                settings=self.settings,
            )
            print(f"  Removed {len(days_to_drop)} oldest day(s): {days_to_drop}")
 
        except Exception as exc:
            print(f"  WARNING: sliding window removal failed ({exc}) — table not trimmed.")
    
    def _retry_failed_days(self) -> list[str]:
        """Re-attempt days that failed all retries in a previous run.

        Successfully recovered days are removed from the failed-days file.
        Returns the list of days that were successfully recovered.
        """
        failed = read_failed_days("failed_istdaten_days", self.settings)
        if not failed:
            return []

        table = self.settings.spark_user_istdaten_table
        recovered: list[str] = []
        still_failed: list[str] = []

        # Only retry days that are not already in the table
        existing_days = self._existing_days_in_table(table)

        for day_str in sorted(failed):
            if day_str in existing_days:
                # Already present (perhaps written partially) — skip
                recovered.append(day_str)
                continue
            success = self._fetch_and_append_with_retry(day_str, table)
            if success:
                recovered.append(day_str)
            else:
                still_failed.append(day_str)

        write_failed_days("failed_istdaten_days", still_failed, self.settings)
        return recovered

    def _existing_days_in_table(self, table: str) -> set[str]:
        rows = self._get_spark().sql(f"SELECT DISTINCT operating_day FROM {table}").collect()
        return {r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]) for r in rows}
    
    def _should_retrain(self) -> bool:
        dates = read_dates("model_trained_dates", self.settings)
        if not dates or "END" not in dates:
            return True
        end = date.fromisoformat(dates["END"])
        return (date.today() - end).days >= self.retrain_interval_days

    def _check_divergence(self) -> tuple[bool, dict]:
        """Evaluate whether the active model has divergenceed on recently arrived data.
 
        Steps:
          1. Read baseline_pinball from divergence_state.json (written after last retrain).
          2. Build a validation period of the most recent divergence_lookback_days days.
          3. Run trainer.evaluate() on that period using the active slot's artifacts.
          4. Extract the q=0.5 pinball loss (same metric as the baseline).
          5. If (current - baseline) / baseline > DIVERGENCE_THRESHOLD → divergence.
 
        Returns (divergenceed: bool, info: dict).
        """
        try:
            state = self._read_divergence_state()
            baseline_pinball = state.get("baseline_pinball")
            if baseline_pinball is None:
                print("  Divergence check skipped — no baseline_pinball in divergence_state.json.")
                return False, {"skipped": "no baseline_pinball"}
 
            slot = state.get("active_slot", "slot_a")
            artifacts = ModelArtifacts(
                base_path=f"{self.settings.local_artifacts_path}/model/{slot}",
                settings=self.settings,
            )
 
            # Validation window: most recent divergence_lookback_days days of istdaten
            istdaten_dates  = read_dates("istdaten_dates", self.settings)
            val_end   = istdaten_dates.get("END", date.today().isoformat())
            val_start = (
                date.fromisoformat(val_end) - timedelta(days=self.divergence_lookback_days)
            ).isoformat()
 
            # Training window: same as what was used at last retrain (needed for
            # correct historical-aggregate feature computation — no label leakage).
            trained_dates = read_dates("model_trained_dates", self.settings)
            train_start   = trained_dates.get("START", self.settings.final_train_start_date)
            train_end     = trained_dates.get("END",   self.settings.final_train_end_date)
 
            training_settings = get_settings(
                istdaten_table=self.settings.spark_user_istdaten_table
            )
            trainer = DelayModelTrainer(
                spark=self._get_spark(),
                settings=training_settings,
                artifacts=artifacts,
            )
            eval_result = trainer.evaluate(
                train_start_date=train_start,
                train_end_date=train_end,
                val_periods=[(val_start, val_end)],
            )
 
            current_pinball = eval_result.get("val_pinball_loss", {}).get("0.5")
            if current_pinball is None:
                print("  Divergence check skipped — q=0.5 pinball not found in eval result.")
                return False, {"skipped": "q=0.5 missing from eval"}
 
            relative_deviation = (current_pinball - baseline_pinball) / baseline_pinball
 
            info = {
                "active_slot":        slot,
                "baseline_pinball":   baseline_pinball,
                "current_pinball":    current_pinball,
                "relative_deviation": relative_deviation,
                "divergence_threshold":    self.divergence_threshold,
                "val_period":         f"{val_start} → {val_end}",
            }
            print(
                f"  Divergence — current_pinball={current_pinball:.4f}, "
                f"baseline_pinball={baseline_pinball:.4f}, "
                f"relative_deviation={relative_deviation:.3f} "
                f"(threshold {self.divergence_threshold})"
            )
 
            divergenceed = relative_deviation > self.divergence_threshold
            return divergenceed, info
 
        except Exception as exc:
            print(f"  Divergence check failed ({exc}) — assuming no divergence.")
            return False, {"error": str(exc)}
    
    # def _append_rows_via_spark(self, rows: list[dict], table: str) -> None:
    #     import pandas as pd
    #     from src.data.istdaten_fetcher import COLUMN_MAP

    #     if not rows:
    #         return
    #     # Enforce the column order defined by COLUMN_MAP so insertInto matches
    #     # the table schema positionally and never silently writes wrong columns.
    #     ordered_cols = list(COLUMN_MAP.values())
    #     pdf = pd.DataFrame(rows)
    #     pdf = pdf[[c for c in ordered_cols if c in pdf.columns]]
    #     spark = self._get_spark()
    #     sdf = spark.createDataFrame(pdf)
    #     sdf.write.insertInto(table)

    def _append_rows_via_spark(self, rows: list[dict], table: str) -> None:
        from src.data.istdaten_fetcher import COLUMN_MAP
        import pandas as pd
        from pyspark.sql import functions as F
        from pyspark.sql.types import DateType, TimestampType, BooleanType, LongType

        if not rows:
            return

        ordered_cols = list(COLUMN_MAP.values())
        pdf = pd.DataFrame(rows)
        pdf = pdf.replace("", None)
        pdf = pdf[[c for c in ordered_cols if c in pdf.columns]]

        spark = self._get_spark()
        sdf = spark.createDataFrame(pdf)

        for field in spark.table(table).schema:
            if field.name not in sdf.columns:
                continue
            if isinstance(field.dataType, DateType):
                sdf = sdf.withColumn(field.name, F.to_date(F.col(field.name)))
            elif isinstance(field.dataType, TimestampType):
                sdf = sdf.withColumn(field.name, F.to_timestamp(F.col(field.name)))
            elif isinstance(field.dataType, BooleanType):
                sdf = sdf.withColumn(field.name, F.col(field.name).cast("boolean"))
            elif isinstance(field.dataType, LongType):
                sdf = sdf.withColumn(field.name, F.col(field.name).cast("long"))

        sdf.writeTo(table).append()

    def _get_spark(self):
        if self.spark is None:
            from src.config.spark_session import get_spark_session
            self.spark = get_spark_session(self.settings, app_name="model-updator")
        return self.spark


