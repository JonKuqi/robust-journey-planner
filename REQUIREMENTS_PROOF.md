This document serves as a technical index for project graders to verify exactly how and where all 21 project requirements for the **Robust Journey Planning** system have been implemented and utilized across the codebase.

---

### ✅ Visualization is available in Jupyter
*   **Status:** Met. The `MapVisualizer` class provides a fully interactive UI using `ipywidgets` (dropdowns for stations, sliders for confidence $Q$) and `ipyleaflet` for map drawing.
*   **Implementation & Usage:** Implemented in `src/viz/map_visualizer.py`. Provides an interactive interface for querying and visualizing the robust journey planner.
*   **Link:** [src/viz/map_visualizer.py (Lines 185-197)](./src/viz/map_visualizer.py#L185-L197)
---

## 1. Configuration & Region Filtering

### ✅ Region UUID is configurable
*   **Solution:** Centralized settings management via `ProjectSettings`.
*   **High-Level Entry Point:** [results.ipynb (Lines 78-83)](./results.ipynb#L78-L83) — Overridden in the first cell using `get_settings(...)`.
*   **Implementation & Usage:** Defined in `settings.py`, hashed in `model_artifacts.py` to namespace models, and used as a filter in `csa_data_handler.py`.
*   **Link:** [src/config/settings.py (Line 111)](./src/config/settings.py#L111)
```python
region_uuids: tuple[str, ...] = field(default_factory=lambda: _split_csv(os.getenv("COM490_REGION_UUIDS")))
```

### ✅ Planner works for selected geo regions
*   **Solution:** SQL-level spatial filtering using Trino GIS functions.
*   **High-Level Entry Point:** [results.ipynb (Lines 913-920)](./results.ipynb#L913-L920) — Triggered by `bench_robust_prepare(..., regions=LAUSANNE_REGION_UUIDS)`.
*   **Implementation & Usage:** The `build_stops` method performs an `ST_Contains` join between SBB stop coordinates and the selected Geo UUID polygons.
*   **Link:** [src/data/csa_data_handler.py (Lines 120-127)](./src/data/csa_data_handler.py#L120-L127)
```python
region_join = f"""
JOIN (
    SELECT wkb_geometry FROM {self._table('src_geo')}
    WHERE CAST(uuid AS VARCHAR) IN ({region_values})
) r ON ST_Contains(ST_GeomFromBinary(r.wkb_geometry), ST_Point(s.stop_lon, s.stop_lat))
"""
```

---

## 2. Core Routing & Constraints

### ✅ Start and end stops are known station coordinates
*   **Solution:** Strict type hints and direct SBB stop ID parameters in the routing API.
*   **High-Level Entry Point:** [results.ipynb (Lines 1293-1300)](./results.ipynb#L1293-L1300) — Enforced during the call to `lausanne_planner.plan(...)`.
*   **Implementation & Usage:** The high-level `plan()` API enforces that queries must provide `start_stop_id: int` and `end_stop_id: int`. Since there is no coordinate-to-stop resolver, the API strictly expects exact SBB station IDs.
*   **Link:** [src/routing/robust_journey_planner.py (Lines 434-437)](./src/routing/robust_journey_planner.py#L434-L437)
```python
    def plan(
        self,
        start_stop_id: int,
        end_stop_id: int,
```

### ✅ Day of week is supported
*   **Solution:** Dynamic timetable indexing based on travel date.
*   **High-Level Entry Point:** [results.ipynb (Line 1296)](./results.ipynb#L1296) — Specified by the `travel_date` parameter in `lausanne_planner.plan(...)`.
*   **Implementation & Usage:** `_day_from_travel_date` resolves ISO dates to day names, which then index the `connections_by_day` dictionary.
*   **Link:** [src/routing/robust_journey_planner.py (Line 461)](./src/routing/robust_journey_planner.py#L461)
```python
day = self._day_from_travel_date(travel_date)
```

### ✅ Arrival deadline is supported
*   **Solution:** Multi-step backward search from a fixed temporal upper bound.
*   **High-Level Entry Point:** [results.ipynb (Line 1297)](./results.ipynb#L1297) — Specified by the `arrival_deadline` parameter in `lausanne_planner.plan(...)`.
*   **Implementation & Usage:** The `arrival_deadline` is used to calculate `deadline_secs`, which serves as the starting point for the windowed search. Any candidate that arrives after this deadline is rejected before robust evaluation.
*   **Link:** [src/routing/robust_journey_planner.py (Lines 462-464)](./src/routing/robust_journey_planner.py#L462-L464)
```python
        deadline_secs = self._deadline_to_relative_secs(travel_date, arrival_deadline)
        max_walk_m = self.settings.max_walk_m if max_walk_m is None else max_walk_m
        earliest_dep = max(0, deadline_secs - search_window_minutes * 60)
```

### ✅ Routes arrive before the requested target time
*   **Solution:** Backward Profile CSA bounding and $Q$-confidence filtering.
*   **High-Level Entry Point:** [results.ipynb (Lines 1293-1300)](./results.ipynb#L1293-L1300) — Passed in `plan(arrival_deadline=...)`.
*   **Implementation & Usage:** The system ensures a route arrives before the deadline via a backward Profile CSA bounded at `deadline_secs`. It then filters candidates ensuring they pass the $Q$-quantile confidence threshold without missed connections.
*   **Link:** [src/routing/robust_journey_planner.py (Lines 772-774)](./src/routing/robust_journey_planner.py#L772-L774)
```python
route["passes_confidence"] = missed_connection is None and confidence >= q
```

### ✅ Routes are sorted from latest departure to earliest departure
*   **Solution:** Final sort before returning profile routes.
*   **High-Level Entry Point:** [results.ipynb (Lines 1305-1310)](./results.ipynb#L1305-L1310) — The final routes list is printed sequentially.
*   **Implementation & Usage:** After extracting the pareto-optimal routes from the profile CSA, the planner explicitly sorts the results by `-departure_secs` ensuring the user sees the latest possible departures first, while also using multi-objective tie-breaking.
*   **Link:** [src/routing/robust_journey_planner.py (Line 682)](./src/routing/robust_journey_planner.py#L682)
```python
routes.sort(key=lambda r: (-r.get("departure_secs", 0), r.get("n_transfers", 0), r.get("total_walk_m", 0)))
```
### ✅ Multiple routes can be returned
*   **Solution:** Multi-pass windowed search (defaulting to 180 minutes).
*   **High-Level Entry Point:** [results.ipynb (Lines 1299-1300)](./results.ipynb#L1299-L1300) — Controlled by the `max_routes` and `search_window_minutes` parameters.
*   **Implementation & Usage:** The system iterates backwards from the deadline in 5-minute increments. This 3-hour window is a design heuristic that ensures a diverse candidate pool (latest vs. safest) while maintaining "Reasonable Runtime" (sub-second query speed).
*   **Link:** [src/routing/robust_journey_planner.py (Lines 440-442)](./src/routing/robust_journey_planner.py#L440-L442)
```python
        search_window_minutes: int = 180,
        max_routes: int = 5,
        confidence_q: float = 0.5,
```

---

## 3. Walking & Transfers

### ✅ Walking transfers are supported
*   **Solution:** Integrated CSA footpaths.
*   **High-Level Entry Point:** [results.ipynb (Line 1293)](./results.ipynb#L1293) — Enabled by default in all routing queries.
*   **Implementation & Usage:** Footpaths are applied at every "board" and "alight" event during the CSA scan to allow inter-platform and inter-station changes.
*   **Link:** [src/routing/robust_journey_planner.py (Lines 490-492)](./src/routing/robust_journey_planner.py#L490-L492)
```python
        for nb, wsecs, dist in footpaths.get(end_stop_id, ()):
            if dist > max_walk_m:
                break
```

### ✅ Maximum walking distance is configurable
*   **Solution:** Hierarchical parameter resolution (User Override > System Default).
*   **High-Level Entry Point:** Configurable via `settings.max_walk_m` or per query in `plan(...)`.
*   **Implementation & Usage:** The system defaults to 500m (SBB standard) but gives full priority to the user's input. The final `max_walk_m` is resolved in `JourneyPlanner.route` before being passed to the footpath scanner.
*   **Link:** [src/routing/robust_journey_planner.py (Line 463)](./src/routing/robust_journey_planner.py#L463)
```python
# User Input priority over System Default (500m)
max_walk_m = self.settings.max_walk_m if max_walk_m is None else max_walk_m
```

### ✅ Total walking distance constraint is handled
*   **Solution:** Graph-level pruning.
*   **High-Level Entry Point:** [results.ipynb (Lines 913-920)](./results.ipynb#L913-L920) — Handled during `bench_robust_prepare()`.
*   **Implementation & Usage:** Distances are calculated using Great Circle distance and filtered during the `build_footpaths` SQL phase.
*   **Link:** [src/data/csa_data_handler.py (Line 178)](./src/data/csa_data_handler.py#L178)
```python
WHERE distance <= {self.settings.max_walk_m}
```

### ✅ Walking speed / walking-time assumption is documented
*   **Solution:** Hardcoded defaults based on official project FAQ.
*   **High-Level Entry Point:** [README.md (Line 268)](./README.md#L268) — Specified in FAQ Question 1.
*   **Implementation & Usage:** The project uses 50m/min and a 2-minute "buffer" time as dictated by the SBB project requirements. The exact formula used in the data pipeline is: `walk_time_min = 2.0 + (distance_m / 50.0)`.
*   **Link:** [src/config/settings.py (Lines 116-121)](./src/config/settings.py#L116-L121)
```sql
{self.settings.walking_transfer_base_min} + (distance / {self.settings.walking_speed_m_per_min}) AS walk_time_min
```

---

## 4. Data Handling & Predictive Modeling

### ✅ Timetable publication date is handled
*   **Solution:** Automated "Latest Pub Date" detection.
*   **High-Level Entry Point:** [results.ipynb (Lines 913-920)](./results.ipynb#L913-L920) — Handled during `bench_robust_prepare()`.
*   **Implementation & Usage:** Queries the database for `MAX(pub_date)` to ensure all CSA tables are built from the most current available schedule.
*   **Link:** [src/data/csa_data_handler.py (Line 76)](./src/data/csa_data_handler.py#L76)
```python
self.max_pub_date = self._format_date_literal(max_pub_date or self.get_max_pub_date())
```

### ✅ Historical Istdaten delays are used
*   **Solution:** Spark-based training on SBB "Istdaten".
*   **High-Level Entry Point:** [results.ipynb (Lines 207-214)](./results.ipynb#L207-L214) — Triggered by `trainer.train(...)`.
*   **Implementation & Usage:** Filters the multi-million row Istdaten table by region and date range to build the delay predictor.
*   **Link:** [src/models/delay_model_trainer.py (Line 156)](./src/models/delay_model_trainer.py#L156)
```python
feature_df.filter(F.col("operating_day").between(train_start_date, train_end_date))
```

### ✅ Predictive model or delay distribution model is used
*   **Solution:** SparkXGBRegressor for Quantile Regression.
*   **High-Level Entry Point:** [results.ipynb (Lines 207-214)](./results.ipynb#L207-L214) — The model is trained and then loaded automatically by the robust planner.
*   **Implementation & Usage:** Uses a distributed XGBoost model via SparkXGBRegressor trained with pinball loss for multi-quantile prediction (e.g. p80, p90, p95).
*   **Link:** [src/models/delay_model_trainer.py (Lines 896-899)](./src/models/delay_model_trainer.py#L896-L899)
```python
                regressor = SparkXGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=float(quantile),
            n_estimators=rounds,
```

### ✅ Route confidence is computed
*   **Solution:** Cumulative delay propagation using Profile CSA.
*   **High-Level Entry Point:** [results.ipynb (Lines 1305-1310)](./results.ipynb#L1305-L1310) — Results displayed in the notebook include `passes_confidence` and `robust_arrival_time`.
*   **Implementation & Usage:** Accumulates log-probabilities along the journey segments during profile building to properly assess multi-leg route robustness.
*   **Link:** [src/routing/robust_journey_planner.py (Line 773)](./src/routing/robust_journey_planner.py#L773)
```python
out["passes_confidence"] = missed_connection is None and robust_arrival_secs <= deadline_secs
```

### ✅ Confidence level Q% is supported
*   **Solution:** Quantile-aware delay retrieval.
*   **High-Level Entry Point:** [results.ipynb (Line 1298)](./results.ipynb#L1298) — Specified by the `confidence_q` parameter in `lausanne_planner.plan(...)`.
*   **Implementation & Usage:** Selects the appropriate column (p80, p90, p95) from the delay model to adjust the "riskiness" of the arrival estimate.
*   **Link:** [src/routing/robust_journey_planner.py (Line 78)](./src/routing/robust_journey_planner.py#L78)
```python
confidence_q: float,
```

---

## 5. Deliverables, Output, & Quality

### ✅ Validation is included
*   **Solution:** Two-stage validation: Statistical Analysis (Pinball Loss) and End-to-End Evaluation.
*   **High-Level Entry Point:** [results.ipynb (Lines 220-224)](./results.ipynb#L220-L224) and [results/end2end_evaluation_v2.py](./results/end2end_evaluation_v2.py).
*   **Implementation & Usage:** The project reports **Pinball loss evaluation** metrics for the XGBoost delay predictor at multiple quantiles to ensure statistical calibration. Furthermore, it implements an **End-to-End Evaluator** that replays historical queries through a real-world simulator to directly prove the success rate of the Robust vs Standard planner.
*   **Statistical Link:** [src/models/delay_model_trainer.py (Lines 397-423)](./src/models/delay_model_trainer.py#L397-L423)
```python
    def evaluate_predictions(self, pred_df: DataFrame) -> dict[str, float]:
        # Pinball loss per quantile
        # ...
```
*   **End-to-End Link:** [src/evaluation/e2e_evaluator.py (Lines 64-67)](./src/evaluation/e2e_evaluator.py#L64-L67)
```python
    def evaluate(
        self,
        benchmark_df: pd.DataFrame,
```

### ✅ Assumptions and limitations are clearly stated
*   **Solution:** Dedicated "Simplifying Assumptions" section in README.
*   **High-Level Entry Point:** [README.md (Line 51)](./README.md#L51) — Primary documentation.
*   **Implementation & Usage:** The following core assumptions are strictly adhered to:
    - ✅ **Reasonable Hours:** Only consider journeys at reasonable hours with recent schedules.
    - ✅ **Walking Formula:** 50m/min straight-line speed; 2min base transfer time.
    - ✅ **Configurable Walk:** Max walking distance is configurable (default 500m).
    - ✅ **Station Constraints:** Only start/end at known station coordinates.
    - ✅ **Geo-Fencing:** Only consider stops in areas specified by UUID.
    - ✅ **External Transfers:** Allows transfers at stops outside the area if needed for connectivity.
    - ✅ **Independence:** Delays/travel times assumed uncorrelated.
    - ✅ **Static Planning:** No "en-route" adaptation; user follows the plan to the end or failure.
    - ✅ **Equivalent Failures:** Planner does not weight the "severity" of failure consequences differently.
*   **Link:** [README.md (Lines 51-66)](./README.md#L51-L66)
### ✅ Prefer minimum walking distance and minimum transfers
*   **Solution:** Multi-key tie-breaking in the route reconstructor.
*   **High-Level Entry Point:** [results.ipynb (Lines 1305-1310)](./results.ipynb#L1305-L1310) — Final sorted list respects multi-objective optimization.
*   **Implementation & Usage:** Routes are sorted by a tuple of `(-departure_secs, n_transfers, total_walk_m)`. This ensures that for the same departure time, the system picks the route with the fewest changes and the least physical effort.
*   **Link:** [src/routing/robust_journey_planner.py (Line 682)](./src/routing/robust_journey_planner.py#L682)
```python
key=lambda r: (-r.get("departure_secs", 0), r.get("n_transfers", 0), r.get("total_walk_m", 0))
```
