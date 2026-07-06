from __future__ import annotations

"""Per-date XGBoost delay lookup generator.

Produces  artifacts/precomputed_day_delay/{date}/delays.json — a flat dict:

    "bpuic__LINE_TEXT__hour" → [q50, q60, q70, q80, q85, q90, q95]  (seconds)

Compared to the global precomputed_delays.json (6-field key, 1M cap, 83 MB):
  - 3-field key: day_of_week, month, and day_type are fixed from the known date
  - Only covers combos present in the day's actual timetable (~50k-150k entries)
  - line_text always uppercased at write time → no case-mismatch with hot loop
  - ~5-15 MB per date, trivially held in memory
"""

import json
import time
from datetime import date as _date
from pathlib import Path

from src.config.settings import ProjectSettings, get_settings
from src.data.calendar_data import CalendarDataHandler
from src.models.delay_model import DelayModel

# Python weekday() → Spark dayofweek (1=Sun, 2=Mon, ..., 7=Sat)
_WEEKDAY_TO_SPARK_DOW = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 1}


class DayDelayGenerator:
    """Generate a per-date XGBoost delay lookup from an already-loaded DelayModel."""

    def __init__(
        self,
        delay_model: DelayModel,
        settings: ProjectSettings | None = None,
    ):
        self.model = delay_model
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        date: str,
        conn_day: dict,
        trip_meta_by_idx: list,
        overwrite: bool = False,
    ) -> str:
        """Build and save the per-date delay lookup.

        Args:
            date:             ISO date string, e.g. "2026-02-05"
            conn_day:         conn[day] struct-of-arrays from RobustJourneyPlanner
                              needs keys: arr_stop, arr_adj, trip_idx
            trip_meta_by_idx: planner.trip_meta_by_idx (list indexed by trip int)
            overwrite:        regenerate even if file already exists

        Returns:
            Absolute path to the saved JSON file.
        """
        out_path = Path(self._day_delay_path(date))
        if out_path.exists() and not overwrite:
            print(f"  already exists, skipping: {out_path}")
            return str(out_path)

        t0 = time.perf_counter()

        # Fixed calendar dimensions for this date
        d         = _date.fromisoformat(date)
        dow       = _WEEKDAY_TO_SPARK_DOW[d.weekday()]
        month     = d.month
        # ZH is the default fallback in DelayModel._get_calendar_features too
        cal       = CalendarDataHandler.get_features_for_date(date, canton_code="ZH")
        day_type  = DelayModel._day_type_from_calendar(cal)
        print(f"  date={date}  dow={dow}  month={month}  day_type={day_type}")

        arr_stop_a = conn_day["arr_stop"]
        arr_adj_a  = conn_day["arr_adj"]
        trip_idx_a = conn_day["trip_idx"]
        n_conns    = len(arr_stop_a)

        # Single pass: collect unique (bpuic, line_text_upper, hour) → transport
        # Transport determines whether a stop gets bus weather features.
        # For a given (bpuic, line_text) the transport mode is always the same.
        seen: dict[tuple[str, str, str], str] = {}
        for idx in range(n_conns):
            ti        = trip_idx_a[idx]
            meta      = trip_meta_by_idx[ti]
            bpuic     = str(arr_stop_a[idx])
            line_text = str(meta.get("line_text") or "UNKNOWN").upper()
            hour      = str(int(arr_adj_a[idx] // 3600 % 24))
            combo     = (bpuic, line_text, hour)
            if combo not in seen:
                seen[combo] = str(meta.get("transport") or "")

        combos    = sorted(seen)
        print(f"  {n_conns} connections → {len(combos)} unique (bpuic, line, hour) combos")

        # Build feature rows reusing DelayModel._build_feature_row for full consistency
        # with training: hist-agg features, weather for buses, calendar flags all included.
        feature_rows: list[dict] = []
        for combo in combos:
            bpuic, line_text, hour = combo
            row = self.model._build_feature_row(
                bpuic=bpuic,
                line_text=line_text,
                transport=seen[combo],
                hour=int(hour),
                day_of_week=dow,
                month=month,
                calendar_features=cal,
            )
            feature_rows.append(row)

        # Batch XGBoost inference — driver only, no Spark
        t_xgb = time.perf_counter()
        all_preds = self.model._predict_xgb(feature_rows)
        q_levels  = sorted(all_preds.keys())
        print(f"  xgb inference: {time.perf_counter() - t_xgb:.1f}s  "
              f"({len(feature_rows)} rows × {len(q_levels)} quantiles)")

        # Enforce monotonicity per combo — models trained independently can cross
        for i in range(len(feature_rows)):
            vals = sorted(all_preds[q][i] for q in q_levels)
            for j, q in enumerate(q_levels):
                all_preds[q][i] = vals[j]

        # Build lookup: key = "bpuic__LINE_TEXT__hour"
        lookup: dict[str, list[float]] = {}
        for i, combo in enumerate(combos):
            bpuic, line_text, hour = combo
            key = f"{bpuic}__{line_text}__{hour}"
            lookup[key] = [round(all_preds[q][i], 2) for q in q_levels]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(lookup), encoding="utf-8")

        size_mb = out_path.stat().st_size / 1e6
        print(f"  saved {len(lookup)} entries → {out_path}  "
              f"({size_mb:.1f} MB, {time.perf_counter() - t0:.1f}s total)")
        return str(out_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _day_delay_path(self, date: str) -> str:
        return (
            f"{self.settings.local_artifacts_path}"
            f"/precomputed_day_delay/{date}/delays.json"
        )

    @staticmethod
    def load_for_date(date: str, settings: ProjectSettings | None = None) -> dict:
        """Load an already-generated day-delay lookup dict into memory.

        Returns an empty dict if the file does not exist yet.
        """
        s    = settings or get_settings()
        path = Path(f"{s.local_artifacts_path}/precomputed_day_delay/{date}/delays.json")
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Standalone test — Feb 5 2026 is a regular Thursday, no Swiss holidays
# ----------------------------------------------------------------------

if __name__ == "__main__":
    TEST_DATE = "2026-02-05"
    TEST_DAY  = "thursday"

    from src.models.model_artifacts import ModelArtifacts
    from src.routing.robust_journey_planner import RobustJourneyPlanner

    print("=" * 50)
    print(f"DayDelayGenerator test — {TEST_DATE} ({TEST_DAY})")
    print("=" * 50)

    # 1. Load the global DelayModel (driver-only, no Spark needed)
    print("\n[1/3] Loading DelayModel from global artifacts...")
    model = DelayModel.load(artifacts=ModelArtifacts())
    print(f"  boosters loaded: {sorted(model.xgb_models.keys())}")
    print(f"  hist_aggs sections: {list(model.hist_aggs.keys())[:5]}")

    # 2. Load planner connections for the target weekday
    #    load_delay_lookup=False: skip delay JSON, we only need CSA arrays
    print(f"\n[2/3] Loading planner connections for {TEST_DAY}...")
    planner = RobustJourneyPlanner()
    planner.prepare(load_delay_lookup=False, require_delay_lookup=False)

    conn_day       = planner.conn[TEST_DAY]
    trip_meta      = planner.trip_meta_by_idx
    print(f"  {len(conn_day['arr_stop'])} connections for {TEST_DAY}")

    # 3. Generate
    print(f"\n[3/3] Generating day delay lookup for {TEST_DATE}...")
    gen  = DayDelayGenerator(delay_model=model)
    path = gen.generate(TEST_DATE, conn_day, trip_meta, overwrite=True)

    # 4. Spot-check the output
    print("\nSpot-check (first 5 entries):")
    lookup = DayDelayGenerator.load_for_date(TEST_DATE)
    for key, vals in list(lookup.items())[:5]:
        print(f"  {key}: {vals}")

    print(f"\nTotal entries in lookup: {len(lookup)}")
    print("Done.")
