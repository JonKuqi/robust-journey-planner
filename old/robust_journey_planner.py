from __future__ import annotations

"""Public robust-journey-planner API."""

import time
from pathlib import Path
from typing import Iterable

from src.config.settings import ProjectSettings, get_settings
from src.models.delay_model import DelayModel
from src.models.delay_model_trainer import DelayModelTrainer
from src.models.model_artifacts import ModelArtifacts
from .confidence_evaluator import ConfidenceEvaluator
from .journey_planner_bacward_scan import JourneyPlanner, _BAR, _THIN_BAR


class RobustJourneyPlanner:
    """Orchestrates scheduled routing, delay prediction, and confidence checks."""

    def __init__(
        self,
        settings: ProjectSettings | None = None,
        journey_planner: JourneyPlanner | None = None,
        delay_model: DelayModel | None = None,
        confidence_evaluator: ConfidenceEvaluator | None = None,
    ):
        """Initialize the robust planner."""
        self.settings = settings or get_settings()
        self.journey_planner = journey_planner
        self.delay_model = delay_model
        self.confidence_evaluator = confidence_evaluator or ConfidenceEvaluator()
        self.prepared = False

    def prepare(
        self,
        regions: Iterable[str] | None = None,
        force_rebuild: bool = False,
        rebuild_csa_prerequisites: bool = True,
        train_delay_model: bool = False,
        force_retrain_delay_model: bool = False,
    ) -> "RobustJourneyPlanner":
        """Prepare scheduled-routing data and load or train the delay model."""
        t_total = time.perf_counter()
        region_uuids = tuple(regions if regions is not None else self.settings.region_uuids)

        # ── CSA routing layer (prints its own breakdown) ──────────────────────
        self.journey_planner = self.journey_planner or JourneyPlanner(settings=self.settings)
        t = time.perf_counter()
        self.journey_planner.prepare(
            regions=region_uuids,
            rebuild=force_rebuild,
            rebuild_prerequisites=rebuild_csa_prerequisites,
        )
        jp_elapsed = time.perf_counter() - t

        # ── model artifacts ───────────────────────────────────────────────────
        print(_BAR)
        print("  ROBUST PLANNER — MODEL LOAD")
        print(_BAR)

        t = time.perf_counter()
        artifacts = ModelArtifacts(settings=self.settings).global_delay_artifacts()
        print(f"  {'model_artifacts.global':<28}  {time.perf_counter() - t:>6.2f}s")

        if train_delay_model:
            t = time.perf_counter()
            DelayModelTrainer(settings=self.settings).train(
                region_uuids=region_uuids,
                force_rebuild_data=force_rebuild,
                force_rebuild_region_stops=force_rebuild or rebuild_csa_prerequisites,
                force_retrain=force_retrain_delay_model or force_rebuild,
            )
            print(f"  {'delay_model_trainer.train':<28}  {time.perf_counter() - t:>6.2f}s")

        if self.delay_model is None:
            if not Path(artifacts.model_metadata_path()).exists():
                print(
                    f"\n  No delay model artifacts found for region '{artifacts.region_key}'.\n"
                    f"  Run prepare(train_delay_model=True) once to train and save it.\n"
                )
                return self
            t = time.perf_counter()
            try:
                self.delay_model = DelayModel.load(artifacts=artifacts)
                print(f"  {'delay_model.load':<28}  {time.perf_counter() - t:>6.2f}s")
            except Exception as exc:
                print(f"\n  Could not load delay model: {exc}\n  Run prepare(train_delay_model=True) to retrain it.\n")
                return self

        print(_THIN_BAR)
        print(f"  {'journey_planner.prepare':<28}  {jp_elapsed:>6.2f}s")
        print(f"  {'TOTAL prepare()':<28}  {time.perf_counter() - t_total:>6.2f}s")
        print(_BAR + "\n")

        self.prepared = True
        return self

    def plan(
        self,
        start_stop_id: int,
        end_stop_id: int,
        travel_date: str,
        arrival_deadline: str,
        confidence_q: float,
        max_routes: int = 5,
        max_walk_m: int | None = None,
        search_window_minutes: int = 180,
        search_step_minutes: int = 5,
    ) -> list[dict]:
        """Return routes that still arrive before the deadline under Q-delay estimates."""
        if not self.prepared:
            raise RuntimeError("RobustJourneyPlanner.prepare() must be called before plan().")
        if self.delay_model is None:
            raise RuntimeError(
                "Delay model was not loaded. Re-run prepare(train_delay_model=True) to train it, "
                "or prepare() if artifacts already exist."
            )

        t_total = time.perf_counter()

        t = time.perf_counter()
        candidate_routes = self.journey_planner.plan_candidates(
            start_stop_id=start_stop_id,
            end_stop_id=end_stop_id,
            travel_date=travel_date,
            arrival_deadline=arrival_deadline,
            max_routes=max_routes * 3,
            max_walk_m=max_walk_m,
            max_probe_window_minutes=search_window_minutes
        )
        
    
        print("="*30)
        print("All candidate routes: ")
        print(candidate_routes)
        print("="*30)
        t_candidates = time.perf_counter() - t

        t = time.perf_counter()
        robust_routes = []
        for route in candidate_routes:
            delayed_route = self.delay_model.add_delays_to_route(route, q=confidence_q)
            evaluated_route = self.confidence_evaluator.route_passes(
                delayed_route,
                arrival_deadline=route.get("arrival_deadline_secs", arrival_deadline),
                confidence_q=confidence_q,
            )
            if evaluated_route["passes_confidence"]:
                robust_routes.append(evaluated_route)
        t_eval = time.perf_counter() - t

        result = self.confidence_evaluator.sort_routes(robust_routes)[:max_routes]

        elapsed = time.perf_counter() - t_total
        print(
            f"  plan                candidates={len(candidate_routes)}  "
            f"plan_candidates={t_candidates*1000:.1f}ms  "
            f"eval_loop={t_eval*1000:.1f}ms  "
            f"total={elapsed*1000:.1f}ms  "
            f"passing={len(result)}"
        )
        return result
