"""End-to-end evaluator: compare standard vs. robust planner against ground truth.

This module takes a benchmark DataFrame (produced by ``JourneyExtractor``) and
replays each query against both planners.  For each query it:

1. Runs the standard planner  (``JourneyPlanner.plan_candidates``)
2. Runs the robust planner    (``RobustJourneyPlanner.plan``)
3. Simulates route execution under real historical delays
4. Computes per-query and aggregate metrics

The output is a results DataFrame suitable for plotting and tabular reporting.

Usage::

    from src.evaluation.e2e_evaluator import E2EEvaluator

    evaluator = E2EEvaluator(
        journey_planner=jp,
        robust_planner=rp,
    )
    results = evaluator.evaluate(benchmark_pdf, q_levels=[0.5, 0.8, 0.9, 0.95])
    evaluator.summary(results)
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

# +
import time
import numpy as np
import pandas as pd
from typing import Optional, List

class E2EEvaluator:
    """Compare standard and robust journey planners against historical ground truth.

    Parameters
    ----------
    journey_planner : JourneyPlanner
        A prepared standard planner instance.
    robust_planner : RobustJourneyPlanner
        A prepared robust planner instance (includes delay model).
    min_transfer_sec : int
        Minimum transfer time (seconds) for feasibility simulation.
    """

    def __init__(
        self,
        journey_planner,
        robust_planner,
        min_transfer_sec: int = 120,
    ):
        self.jp = journey_planner
        self.rp = robust_planner
        self.min_transfer_sec = min_transfer_sec

    def evaluate(
        self,
        benchmark_pdf: pd.DataFrame,
        q_levels: Optional[List[float]] = None,
        max_routes: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run the full evaluation loop."""
        if q_levels is None:
            q_levels = [0.5, 0.8, 0.9, 0.95]
        print("Sorting by operating day")
        benchmark_pdf = benchmark_pdf.sort_values(by="operating_day").reset_index(drop=True)
        results = []
        total = len(benchmark_pdf)

        for idx, row in benchmark_pdf.iterrows():
            if verbose and (idx % 10 == 0):
                print(f"  Evaluating query {idx+1}/{total} ...")

            origin = int(row["origin_stop_id"])
            dest = int(row["dest_stop_id"])
            day = str(row["operating_day"])
            deadline_secs = int(row["query_deadline_secs"])
            deadline_str = self._secs_to_time(deadline_secs)

            # Ground truth from benchmark
            gt = self._extract_ground_truth(row)

            # ── Standard planner ─────────────────────────────────────────
            try:
                t0 = time.perf_counter()
                std_routes = self.jp.plan(
                    start_stop_id=origin,
                    end_stop_id=dest,
                    travel_date=day,
                    arrival_deadline=deadline_str,
                    max_routes=max_routes,
                    confidence_q=0.0,
                )
                std_time_ms = (time.perf_counter() - t0) * 1000
            except Exception as e:
                std_routes = []
                std_time_ms = 0
                if verbose:
                    print(f"    ⚠ Standard planner error: {e}")

            # Evaluate standard routes against ground truth
            std_eval = self._evaluate_routes(
                routes=std_routes,
                ground_truth=gt,
                label="standard",
            )

            # ── Bake Delays for Robust Planner (if needed) ───────────────
            if getattr(self, "_last_baked_day", None) != day:
                try:
                    self.rp._bake_quantiles_for_date(day)
                    self._last_baked_day = day
                except Exception as e:
                    if verbose:
                        print(f"    ⚠ Could not bake quantiles for {day}: {e}")

            # ── Robust planner (for each Q level) ────────────────────────
            for q in q_levels:
                try:
                    t0 = time.perf_counter()
                    rob_routes = self.rp.plan(
                        start_stop_id=origin,
                        end_stop_id=dest,
                        travel_date=day,
                        arrival_deadline=deadline_str,
                        confidence_q=q,
                        max_routes=max_routes,
                    )
                    rob_time_ms = (time.perf_counter() - t0) * 1000
                except Exception as e:
                    rob_routes = []
                    rob_time_ms = 0
                    if verbose:
                        print(f"    ⚠ Robust planner (Q={q}) error: {e}")

                rob_eval = self._evaluate_routes(
                    routes=rob_routes,
                    ground_truth=gt,
                    label="robust",
                )

                # ── Build result row ─────────────────────────────────────
                result_row = {
                    "query_idx": idx,
                    "origin_stop_id": origin,
                    "dest_stop_id": dest,
                    "operating_day": day,
                    "query_deadline_secs": deadline_secs,
                    "deadline_buffer_sec": int(row.get("deadline_buffer_sec", 0)),
                    "num_transfers_gt": int(row.get("num_transfers", 0)),
                    "journey_completed_gt": bool(row.get("journey_completed", True)),
                    "dest_delay_sec_gt": float(row.get("dest_delay_sec", 0)),
                    "stratum": str(row.get("stratum", "")),
                    "time_of_day": str(row.get("time_of_day", "")),
                    "day_type": str(row.get("day_type", "")),
                    "transfer_tightness": str(row.get("transfer_tightness", "")),
                    "mode_mix": str(row.get("mode_mix", "")),
                    "delay_category": str(row.get("delay_category", "")),
                    "confidence_q": q,
                    "std_n_routes": len(std_routes),
                    "std_time_ms": std_time_ms,
                    **{f"std_{k}": v for k, v in std_eval.items()},
                    "rob_n_routes": len(rob_routes),
                    "rob_time_ms": rob_time_ms,
                    **{f"rob_{k}": v for k, v in rob_eval.items()},
                    "routes_identical": self._routes_identical(std_routes, rob_routes),
                    "robust_proposes_earlier": self._robust_proposes_earlier(std_routes, rob_routes),
                    "robust_filters_routes": len(std_routes) > len(rob_routes),
                }

                # Path overlap with ground truth
                gt_trip_ids = set(gt.get("trip_ids", []))
                if gt_trip_ids:
                    std_trips = self._extract_trip_ids(std_routes)
                    rob_trips = self._extract_trip_ids(rob_routes)
                    result_row["std_path_overlap"] = self._jaccard(gt_trip_ids, std_trips)
                    result_row["rob_path_overlap"] = self._jaccard(gt_trip_ids, rob_trips)
                else:
                    result_row["std_path_overlap"] = None
                    result_row["rob_path_overlap"] = None

                results.append(result_row)

        return pd.DataFrame(results)

    # ── Evaluation helpers ──────────────────────────────────────────────────

    def _extract_ground_truth(self, row: pd.Series) -> dict:
        """Extract ground truth journey details from a benchmark row."""
        gt = {
            "origin_stop_id": int(row["origin_stop_id"]),
            "dest_stop_id": int(row["dest_stop_id"]),
            "dest_scheduled_arr_secs": int(row.get("dest_scheduled_arr_secs", 0)),
            "dest_actual_arr_secs": int(row.get("dest_actual_arr_secs", 0)),
            "dest_delay_sec": float(row.get("dest_delay_sec", 0)),
            "journey_completed": bool(row.get("journey_completed", True)),
            "num_transfers": int(row.get("num_transfers", 0)),
            "trip_ids": [],
            "legs": [],
            "transfers": [],
        }

        # Collect per-leg trip IDs and delay info
        for i in range(1, 5):
            tid = row.get(f"leg{i}_trip_id")
            if tid is not None and pd.notna(tid):
                gt["trip_ids"].append(str(tid))
                gt["legs"].append({
                    "trip_id": str(tid),
                    "board_stop_id": row.get(f"leg{i}_board_stop_id"),
                    "alight_stop_id": row.get(f"leg{i}_alight_stop_id"),
                    "arr_delay_sec": float(row.get(f"leg{i}_arr_delay_sec", 0) or 0),
                    "dep_delay_sec": float(row.get(f"leg{i}_dep_delay_sec", 0) or 0),
                    "line_text": str(row.get(f"leg{i}_line_text", "")),
                    "mode": str(row.get(f"leg{i}_mode", "")),
                })

            # Collect transfer info
            xfer_made = row.get(f"xfer{i}_made")
            if xfer_made is not None and pd.notna(xfer_made):
                gt["transfers"].append({
                    "at_stop_id": row.get(f"xfer{i}_stop_id"),
                    "scheduled_headway_sec": float(row.get(f"xfer{i}_scheduled_headway_sec", 0) or 0),
                    "actual_headway_sec": float(row.get(f"xfer{i}_actual_headway_sec", 0) or 0),
                    "made": bool(xfer_made),
                })

        return gt

    def _evaluate_routes(
        self,
        routes: list[dict],
        ground_truth: dict,
        label: str,
    ) -> dict:
        """Evaluate a set of planner routes against ground truth."""
        if not routes:
            return {
                "found_route": False,
                "best_departure_secs": None,
                "best_arrival_secs": None,
                "best_n_transfers": None,
                "best_walk_m": None,
                "best_duration_sec": None,
                "would_arrive_on_time": None,
                "simulated_success": None,
            }

        best = routes[0]

        # Simulate: would this route succeed under real delays?
        simulated = self._simulate_route_under_real_delays(best, ground_truth)

        deadline_secs = ground_truth.get("dest_scheduled_arr_secs", 0)

        return {
            "found_route": True,
            "best_departure_secs": best.get("departure_secs"),
            "best_arrival_secs": best.get("arrival_secs"),
            "best_n_transfers": best.get("n_transfers", 0),
            "best_walk_m": best.get("total_walk_m", 0),
            "best_duration_sec": best.get("duration_sec", 0),
            "would_arrive_on_time": best.get("arrival_secs", 0) <= deadline_secs,
            "simulated_success": simulated["success"],
        }

    def _simulate_route_under_real_delays(
        self,
        route: dict,
        ground_truth: dict,
    ) -> dict:
        """Simulate whether a proposed route would complete under real delays."""
        steps = route.get("steps", [])
        gt_delays = {
            leg["trip_id"]: leg["arr_delay_sec"]
            for leg in ground_truth.get("legs", [])
        }
        avg_delay = np.mean(list(gt_delays.values())) if gt_delays else 0

        cumulative_delay = 0.0
        success = True

        ride_steps = [(i, s) for i, s in enumerate(steps) if s.get("type") == "ride"]

        for step_idx, (i, step) in enumerate(ride_steps):
            trip_id = step.get("trip_id", "")

            # Apply real delay if we know this trip, else use average
            real_delay = gt_delays.get(trip_id, avg_delay)
            adjusted_arrival = step.get("arrival_secs", 0) + real_delay + cumulative_delay

            # Check if there's a next ride step (= transfer required)
            if step_idx + 1 < len(ride_steps):
                _, next_ride = ride_steps[step_idx + 1]
                next_dep = next_ride.get("departure_secs", 0)

                # Account for walk time between
                walk_time = 0
                for j in range(i + 1, len(steps)):
                    if steps[j].get("type") == "walk":
                        walk_time += steps[j].get("duration_sec", 0)
                    else:
                        break

                available = next_dep - adjusted_arrival - walk_time
                if available < self.min_transfer_sec:
                    success = False
                    break

                cumulative_delay = max(0, adjusted_arrival - step.get("arrival_secs", 0))

        return {"success": success}

    # ── Comparison helpers ──────────────────────────────────────────────────

    @staticmethod
    def _routes_identical(std_routes: list, rob_routes: list) -> bool:
        """Check if two route lists produce the same trip sequences."""
        if len(std_routes) != len(rob_routes):
            return False
        for s, r in zip(std_routes, rob_routes):
            s_trips = tuple(
                step.get("trip_id") for step in s.get("steps", [])
                if step.get("type") == "ride"
            )
            r_trips = tuple(
                step.get("trip_id") for step in r.get("steps", [])
                if step.get("type") == "ride"
            )
            if s_trips != r_trips:
                return False
        return True

    @staticmethod
    def _robust_proposes_earlier(std_routes: list, rob_routes: list) -> bool:
        """Check if the robust planner's best route departs earlier."""
        if not std_routes or not rob_routes:
            return False
        std_dep = std_routes[0].get("departure_secs", 0)
        rob_dep = rob_routes[0].get("departure_secs", 0)
        return rob_dep < std_dep

    @staticmethod
    def _extract_trip_ids(routes: list) -> set:
        """Collect all trip IDs from a list of routes."""
        trip_ids = set()
        for route in routes:
            for step in route.get("steps", []):
                if step.get("type") == "ride" and step.get("trip_id"):
                    trip_ids.add(step["trip_id"])
        return trip_ids

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        """Jaccard similarity between two sets."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    @staticmethod
    def _secs_to_time(secs: int) -> str:
        """Convert seconds-since-midnight to HH:MM:SS."""
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ── Aggregate reporting ─────────────────────────────────────────────────

    @staticmethod
    def summary(results_df: pd.DataFrame) -> pd.DataFrame:
        """Produce aggregate metrics grouped by confidence level."""
        agg = results_df.groupby("confidence_q").agg(
            n_queries=("query_idx", "count"),
            std_recovery_rate=("std_found_route", "mean"),
            rob_recovery_rate=("rob_found_route", "mean"),
            std_success_rate=("std_simulated_success", lambda x: x.dropna().mean()),
            rob_success_rate=("rob_simulated_success", lambda x: x.dropna().mean()),
            routes_identical_pct=("routes_identical", "mean"),
            robust_proposes_earlier_pct=("robust_proposes_earlier", "mean"),
            robust_filters_pct=("robust_filters_routes", "mean"),
            std_avg_time_ms=("std_time_ms", "mean"),
            rob_avg_time_ms=("rob_time_ms", "mean"),
            std_path_overlap=("std_path_overlap", lambda x: x.dropna().mean()),
            rob_path_overlap=("rob_path_overlap", lambda x: x.dropna().mean()),
        ).reset_index()
        return agg

    @staticmethod
    def summary_by_stratum(results_df: pd.DataFrame, stratum_col: str = "stratum") -> pd.DataFrame:
        """Produce aggregate metrics grouped by stratification dimension and Q."""
        agg = results_df.groupby([stratum_col, "confidence_q"]).agg(
            n_queries=("query_idx", "count"),
            std_success_rate=("std_simulated_success", lambda x: x.dropna().mean()),
            rob_success_rate=("rob_simulated_success", lambda x: x.dropna().mean()),
            routes_identical_pct=("routes_identical", "mean"),
            robust_filters_pct=("robust_filters_routes", "mean"),
        ).reset_index()
        return agg

    @staticmethod
    def calibration_table(results_df: pd.DataFrame) -> pd.DataFrame:
        """Check calibration: at each Q level, is the robust success rate ≈ Q?"""
        cal = results_df.groupby("confidence_q").agg(
            rob_empirical_success=("rob_simulated_success", lambda x: x.dropna().mean()),
            std_empirical_success=("std_simulated_success", lambda x: x.dropna().mean()),
            n_queries=("query_idx", "count"),
        ).reset_index()

        cal["calibration_gap"] = cal["rob_empirical_success"] - cal["confidence_q"]
        return cal

