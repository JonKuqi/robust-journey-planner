# End-to-End Journey Evaluation V2

This folder contains the complete, scientifically rigorous evaluation pipeline comparing the standard CSA (Connection Scan Algorithm) journey planner against the delay-aware (robust) journey planner. 

This V2 iteration significantly improves upon previous evaluation methodologies by utilizing real historical multi-leg journeys, incorporating comprehensive stratification, and accurately re-simulating ground-truth delays.

## 📁 Pipeline Components

The evaluation is driven by three main components, coordinated by the main Jupyter notebook script:

### 1. Journey Extractor (`src/evaluation/journey_extractor.py`)

The `JourneyExtractor` mines the SBB `istdaten` Iceberg table to construct a realistic historical benchmark. 

**Key Responsibilities:**
- Extracts direct (0 transfers) and complex journeys (up to 3 transfers).
- Discovers transfers at **any walkable stop pair** (mimicking real CSA capabilities, not just major hubs).
- Computes comprehensive stratification dimensions: transfer tightness, time-of-day, mode mix, day types (weekday/weekend/holiday), and delay categories.
- Introduces variable deadline buffers (e.g., exactly on time, +10 mins, +30 mins) to stress-test the robust planner.
- Output: A highly-stratified Parquet dataset (`e2e_benchmark_v2.parquet`) containing benchmark queries, ready to be fed into the evaluators.

[View Journey Extractor Flowchart](flowcharts/journey_extractor_flowchart.md)

### 2. End-to-End Evaluator (`src/evaluation/e2e_evaluator.py`)

The `E2EEvaluator` takes the extracted benchmark and replays each query against both the standard and robust journey planners.

**Key Responsibilities:**
- Queries the Standard Planner (`JourneyPlanner.plan_candidates`).
- Queries the Robust Planner (`RobustJourneyPlanner.plan`) at varying confidence levels (e.g., $Q \in \{0.5, 0.8, 0.9, 0.95\}$).
- **Delay Simulation**: Re-simulates the proposed routes against the exact historical delays observed in the ground truth data. If a connection is missed due to historical delays, the simulated route fails.
- Computes granular metrics per query:
  - Route identity (did both planners suggest the exact same route?)
  - Simulated success rates.
  - Path overlap (Jaccard similarity to the actual ground-truth route).
  - Timing and overhead performance.

[View E2E Evaluator Flowchart](flowcharts/e2e_evaluator_flowchart.md)

### 3. Main Evaluation Script (`results/end2end_evaluation_v2.py`)

This PySpark script orchestrates the entire process on the cluster, performing data extraction, running the evaluation, and rendering the final analytical comparisons.

**Workflow Phases:**
1. **Setup & Data Extraction**: Defines test days across different calendars (weekdays, weekends, holidays) and runs `JourneyExtractor`.
2. **Prepare Planners**: Initializes and builds caches for `JourneyPlanner` and `RobustJourneyPlanner`.
3. **Run Evaluation**: Executes the `E2EEvaluator` logic over all benchmark rows.
4. **Analysis & Visualization**: Groups results by varying stratifications and visualizes the comparisons (plots saved to `figs/`):
   - **Calibration**: Verifies if the requested robust confidence level matches the simulated empirical success rate.
   - **Standard vs Robust**: Compares success rates and route discrepancies across $Q$ levels.
   - **Transfer Tightness Breakdown**: Showcases the robust planner's ability to filter unreliable connections on tight headways (< 5 mins).
   - **Deadline Buffer Impact**: Visualizes differences under varying schedule tolerances.
   - **Path Overlap & Timing**: Compares planners based on historical path reconstruction and latency.

## 📊 Evaluation Improvements (V2 vs V1)

| Aspect | V1 (Old) | V2 (New) |
| :--- | :--- | :--- |
| **Route Paths** | Origin/Destination only | Full per-leg trip IDs, intermediate stops, exact times |
| **Transfers** | Only 18 major hubs | At any stop (True CSA capability) |
| **Transfer Tightness** | 5–30 min only | Includes extremely tight (2–5 min) connections |
| **Deadline** | Strictly scheduled arrival | Parametrizable buffers: +0, +10, +30 min tolerances |
| **Delay Data** | Last-leg delay only | Per-leg actual delays + full transfer feasibility checks |
| **Stratification** | Limited (Transfers $\times$ delay) | Multi-dimensional (Transfers, tightness, time-of-day, mode mix, outcome) |
| **Metrics** | None defined | Success rate, planner calibration, path overlap, overhead timing |
| **$Q$ Variation** | Not tested | Parameter sweep $Q \in \{0.5, 0.8, 0.9, 0.95\}$ |
