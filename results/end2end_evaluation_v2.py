# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # End-to-End Evaluation V2: Robust vs Standard Journey Planner
#
# This notebook implements a **scientifically rigorous** end-to-end evaluation
# comparing the standard CSA journey planner against the robust (delay-aware)
# journey planner.
#
# ## Key Improvements Over V1
#
# | Aspect | V1 (Old) | V2 (New) |
# |:-------|:---------|:---------|
# | **Route paths** | Only origin/dest stored | Full per-leg trip IDs, stops, times |
# | **Transfers** | Only at 18 major hubs | At any stop (CSA capability) |
# | **Transfer tightness** | Only 5–30 min | Includes tight (2–5 min) connections |
# | **Deadline** | = scheduled arrival | Varies: +0, +10, +30 min buffer |
# | **Delay data** | Last-leg delay only | Per-leg actual delays + transfer feasibility |
# | **Stratification** | 8 levels (transfers × delay) | Multi-dimensional: transfers, tightness, time-of-day, mode, outcome |
# | **Metrics** | None defined | Success rate, calibration, path overlap, timing |
# | **Q variation** | Not tested | Sweeps Q ∈ {0.5, 0.8, 0.9, 0.95} |

# %% [markdown]
# ## Phase 1: Setup & Data Extraction

# %%
import os
import sys

# Add project root to path so 'src' can be imported
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pwd
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from random import randrange
import pyspark.sql.functions as F
from pyspark.sql.window import Window

username = pwd.getpwuid(os.getuid()).pw_name
hadoopFS = os.getenv('HADOOP_FS', None)
groupName = "J1"

userns = f"iceberg.{username}_iceberg"
sharedns = 'iceberg.com490_iceberg'
groupfs = f"{hadoopFS}/user/groups/com-490/{groupName}"

print(f"hadoopFS={hadoopFS}")
print(f"userns={userns}")

# %%
spark = (SparkSession
            .builder
            .appName(username + '-e2e-evaluation-v2')
            .config('spark.ui.port', randrange(4050, 4450, 5))
            .config("spark.executorEnv.PYTHONPATH", ":".join(sys.path))
            .config('spark.jars',
                    f'{hadoopFS}/data/com-490/jars/iceberg-spark-runtime-3.5_2.13-1.6.1.jar,'
                    f'{hadoopFS}/data/com-490/jars/sedona-spark-shaded-3.5_2.13-1.7.1.jar,'
                    f'{hadoopFS}/data/com-490/jars/geotools-wrapper-1.7.1-28.5.jar'
            )
            .config('spark.sql.extensions', 'org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions')
            .config('spark.sql.catalog.iceberg', 'org.apache.iceberg.spark.SparkCatalog')
            .config('spark.sql.catalog.iceberg.type', 'hadoop')
            .config('spark.sql.catalog.iceberg.warehouse', f'{hadoopFS}/data/com-490/silver/')
            .config('spark.sql.catalog.spark_catalog', 'org.apache.iceberg.spark.SparkSessionCatalog')
            .config('spark.sql.catalog.spark_catalog.type', 'hadoop')
            .config('spark.sql.catalog.spark_catalog.warehouse', f'{hadoopFS}/user/{username}/assignment-3/warehouse')
            .config("spark.sql.warehouse.dir", f'{hadoopFS}/user/{username}/assignment-3/spark/warehouse')
            .config("spark.executor.memory", "6g")
            .config("spark.executor.cores", "4")
            .config("spark.executor.instances", "4")
        ).master('yarn').getOrCreate()

spark.sparkContext

# %% [markdown]
# ### Step 1.1: Extract Historical Journey Benchmark
#
# We mine the `istdaten` table for real multi-leg journeys with full
# per-leg details.  Transfers are discovered at **any stop** (not just
# major hubs), and we include tight connections (2–5 min) that stress-test
# the delay model.

# %%
from src.evaluation.journey_extractor import JourneyExtractor

# ── Configure test days across multiple day types ──
testing_calendar = {
    "weekday": [
        "2025-11-12",  # Wednesday
        "2025-12-10",  # Wednesday
        "2026-01-20",  # Tuesday
    ],
    "weekend": [
        "2025-11-15",  # Saturday
        "2025-12-14",  # Sunday
        "2026-01-24",  # Saturday
    ],
    "holiday": [
        "2025-12-25",  # Christmas
        "2025-12-31",  # New Year's Eve
        "2026-01-02",  # Post-New Year
    ],
}

# Flatten all test days
all_test_days = [day for days in testing_calendar.values() for day in days]
print(f"📅 Total test days: {len(all_test_days)}")
print(f"   Days: {all_test_days}")

# %%
# ── Run the extraction ── takes 5 hours 
extractor = JourneyExtractor(
    spark=spark,
    istdaten_table="iceberg.sbb.istdaten",
    min_transfer_sec=120,   # Include tight 2-min transfers
    max_transfer_sec=2700,  # Up to 45 min
)

benchmark_df = extractor.extract(
    test_days=all_test_days,
    samples_per_stratum=30,
    max_transfers=3,
    seed=42,
    deadline_buffer_options_sec=[0, 600, 1800],  # 0, 10 min, 30 min buffers
)

print(f"\n📊 Benchmark shape: {benchmark_df.count()} rows")
# ── Save to HDFS ──
save_path = f"{hadoopFS}/user/{username}/e2e_benchmark_v2.parquet"
benchmark_df.write.format("parquet").mode("overwrite").save(save_path)
print(f"✅ Benchmark saved to: {save_path}")

# %%

# %% [markdown]
# ### Step 1.2: Benchmark Statistics
#
# Let's verify the benchmark quality: stratification balance, transfer
# tightness distribution, mode mix, and day type coverage.

# %%
# Reload (or use from memory)
save_path = f"{hadoopFS}/user/{username}/e2e_benchmark_v2.parquet"
benchmark_df = spark.read.parquet(save_path)
benchmark_pdf = benchmark_df.toPandas()

print(f"📊 Total evaluation queries: {len(benchmark_pdf)}")
print(f"\n{'='*60}")
print("Distribution by stratum:")
print(benchmark_pdf['stratum'].value_counts().sort_index().to_string())

# %%
print(f"\n{'='*60}")
print("Distribution by num_transfers:")
print(benchmark_pdf['num_transfers'].value_counts().sort_index().to_string())

print(f"\n{'='*60}")
print("Distribution by transfer_tightness:")
print(benchmark_pdf['transfer_tightness'].value_counts().sort_index().to_string())

print(f"\n{'='*60}")
print("Distribution by time_of_day:")
print(benchmark_pdf['time_of_day'].value_counts().sort_index().to_string())

print(f"\n{'='*60}")
print("Distribution by day_type:")
print(benchmark_pdf['day_type'].value_counts().sort_index().to_string())

print(f"\n{'='*60}")
print("Distribution by journey_completed (ground truth):")
print(benchmark_pdf['journey_completed'].value_counts().to_string())

print(f"\n{'='*60}")
print("Distribution by mode_mix:")
print(benchmark_pdf['mode_mix'].value_counts().to_string())

print(f"\n{'='*60}")
print("Distribution by delay_category:")
print(benchmark_pdf['delay_category'].value_counts().to_string())

print(f"\n{'='*60}")
print("Distribution by deadline_buffer_sec:")
print(benchmark_pdf['deadline_buffer_sec'].value_counts().sort_index().to_string())

# %%
# Preview a few rows
print("\n👀 Sample benchmark rows:")
display(benchmark_pdf[['operating_day', 'origin_stop_name', 'dest_stop_name',
                        'num_transfers', 'transfer_tightness', 'time_of_day',
                        'journey_completed', 'dest_delay_sec', 'deadline_buffer_sec',
                        'query_deadline_secs']].head(20))

# %% [markdown]
# ## Phase 2: Prepare Planners
#
# We prepare both the standard and robust journey planners for the
# evaluation region.

# %%
import importlib
from src.routing import robust_journey_planner

# 1. Force Python to reload the file I just edited on disk!
importlib.reload(robust_journey_planner)

# 2. Now import the class from the newly loaded file
from src.routing.robust_journey_planner import RobustJourneyPlanner
from src.config.settings import get_settings

# Use your configured region UUIDs (all of Switzerland, or specific regions)
settings = get_settings()

# ── Standard Planner (0-buffer CSA) ──
# We tell it NOT to load the delay models, so it acts exactly like the old JourneyPlanner
jp = RobustJourneyPlanner(settings=settings)
jp.prepare(
    rebuild=False, 
    rebuild_prerequisites=False,
    load_delay_lookup=False  # <--- This forces standard mode
)

# ── Robust Planner ──
# We tell it TO load the delay models for the robust confidence routing
rp = RobustJourneyPlanner(settings=settings)
rp.prepare(
    rebuild=False, 
    rebuild_prerequisites=False,
    load_delay_lookup=True  # <--- This enables robust mode
)

print("✅ Both planners prepared.")


# %% [markdown]
# ## Phase 3: Run Evaluation
#
# We replay each benchmark query against both planners across multiple
# confidence levels.  For each query we:
#
# 1. Run the **standard planner** → get scheduled routes
# 2. Run the **robust planner** at Q ∈ {0.5, 0.8, 0.9, 0.95} → get filtered routes
# 3. **Simulate** each proposed route under real historical delays
# 4. Compare: success rates, route differences, calibration

# %%
# 1. Import the module itself
from src.evaluation import e2e_evaluator
import importlib
# 2. Reload the module
importlib.reload(e2e_evaluator)
from src.evaluation.e2e_evaluator import E2EEvaluator
import time

# ── CONFIGURATION ──
QUICK_TEST = True  # Set to False to run the full benchmark!

if QUICK_TEST:
    print("⚠️  QUICK_TEST is ON: Sampling 20 queries for a fast test run.")
    eval_pdf = benchmark_pdf.sample(n=20, random_state=42)
else:
    eval_pdf = benchmark_pdf

evaluator = E2EEvaluator(
    journey_planner=jp,
    robust_planner=rp,
    min_transfer_sec=120,
)

# Clear any cached routes from planner setup
# rp.clear_cache()

Q_LEVELS = [0.50, 0.80, 0.90, 0.95]

print(f"🚀 Starting evaluation on {len(eval_pdf)} queries × {len(Q_LEVELS)} Q levels ...")
print(f"   = {len(eval_pdf) * len(Q_LEVELS)} total evaluations")

t0 = time.perf_counter()
results_df = evaluator.evaluate(
    benchmark_pdf=eval_pdf,  # We use the filtered dataframe here
    q_levels=Q_LEVELS,
    max_routes=5,
    verbose=True,
)
elapsed = time.perf_counter() - t0
print(f"\n✅ Evaluation complete in {elapsed:.1f}s ({len(results_df)} result rows)")

# %%
import numpy as np

results_df_for_spark = results_df.dropna(axis=1, how='all')

# 3. Save it safely to your cluster
if QUICK_TEST:
    results_save_path = f"{hadoopFS}/user/{username}/e2e_results_v2_TEST.parquet"
else:
    results_save_path = f"{hadoopFS}/user/{username}/e2e_results_v2.parquet"

results_spark_df = spark.createDataFrame(results_df_for_spark)
results_spark_df.write.format("parquet").mode("overwrite").save(results_save_path)
print(f"✅ Results safely saved to: {results_save_path}")

# 4. Now display your metrics!
print("📊 OVERALL SUMMARY")
display(E2EEvaluator.summary(results_df))

print("\n📈 CALIBRATION TABLE")
display(E2EEvaluator.calibration_table(results_df))



# %% [markdown]
# ## Phase 4: Analysis & Visualization
#
# ### 4.1 Overall Summary by Confidence Level

# %%
summary = evaluator.summary(results_df)
print("📊 Overall Summary by Confidence Level:")
print("="*80)
display(summary)

# %% [markdown]
# ### 4.2 Calibration Check
#
# A well-calibrated robust planner should have:
# - At Q=0.50 → ~50% empirical success rate
# - At Q=0.90 → ~90% empirical success rate
#
# If the robust planner always succeeds regardless of Q, the model
# isn't contributing useful predictions (this was the V1 bug).

# %%
cal = evaluator.calibration_table(results_df)
print("📊 Calibration Table:")
display(cal)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
ax.plot(cal['confidence_q'], cal['rob_empirical_success'], 'bo-', markersize=10, label='Robust planner')
ax.plot(cal['confidence_q'], cal['std_empirical_success'], 'rs--', markersize=8, label='Standard planner')
ax.set_xlabel('Requested Confidence Q', fontsize=12)
ax.set_ylabel('Empirical Success Rate', fontsize=12)
ax.set_title('Planner Calibration: Requested vs. Achieved Confidence', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.4, 1.0)
ax.set_ylim(0.0, 1.05)
plt.tight_layout()
os.makedirs('figs', exist_ok=True)
plt.savefig('figs/e2e_calibration.png', dpi=150)
plt.show()

# %% [markdown]
# ### 4.3 Success Rate: Standard vs Robust

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: By Q level
ax = axes[0]
x = np.arange(len(Q_LEVELS))
width = 0.35
ax.bar(x - width/2, summary['std_success_rate'], width, label='Standard', color='#2196F3', alpha=0.8)
ax.bar(x + width/2, summary['rob_success_rate'], width, label='Robust', color='#4CAF50', alpha=0.8)
ax.set_xlabel('Confidence Level Q')
ax.set_ylabel('Simulated Success Rate')
ax.set_title('On-Time Success Rate by Q Level')
ax.set_xticks(x)
ax.set_xticklabels([f'Q={q}' for q in Q_LEVELS])
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')

# Right: Route identity (are they the same?)
ax = axes[1]
ax.bar(x, summary['routes_identical_pct'], color='#FF9800', alpha=0.8)
ax.set_xlabel('Confidence Level Q')
ax.set_ylabel('Fraction of Identical Routes')
ax.set_title('Route Identity: % Same Routes (std vs robust)')
ax.set_xticks(x)
ax.set_xticklabels([f'Q={q}' for q in Q_LEVELS])
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figs/e2e_success_comparison.png', dpi=150)
plt.show()

# %% [markdown]
# ### 4.4 Breakdown by Transfer Tightness
#
# This is where the robust planner should shine: for tight transfers
# (< 5 min headway), the delay model should filter out unreliable routes.

# %%
tight_results = evaluator.summary_by_stratum(results_df, stratum_col="transfer_tightness")
print("📊 Results by Transfer Tightness:")
display(tight_results)

fig, ax = plt.subplots(figsize=(10, 6))
for q in Q_LEVELS:
    q_data = tight_results[tight_results['confidence_q'] == q]
    ax.plot(q_data['transfer_tightness'], q_data['rob_success_rate'],
            'o-', label=f'Robust Q={q}', markersize=8)

# Add standard planner line (Q-independent)
q_any = Q_LEVELS[0]
std_data = tight_results[tight_results['confidence_q'] == q_any]
ax.plot(std_data['transfer_tightness'], std_data['std_success_rate'],
        'ks--', label='Standard', markersize=10, linewidth=2)

ax.set_xlabel('Transfer Tightness', fontsize=12)
ax.set_ylabel('Simulated Success Rate', fontsize=12)
ax.set_title('Success Rate by Transfer Tightness', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig('figs/e2e_tightness_breakdown.png', dpi=150)
plt.show()

# %% [markdown]
# ### 4.5 Breakdown by Number of Transfers

# %%
xfer_results = evaluator.summary_by_stratum(results_df, stratum_col="num_transfers_gt")
print("📊 Results by Number of Transfers:")
display(xfer_results)

# %% [markdown]
# ### 4.6 Breakdown by Time of Day

# %%
tod_results = evaluator.summary_by_stratum(results_df, stratum_col="time_of_day")
print("📊 Results by Time of Day:")
display(tod_results)

# %% [markdown]
# ### 4.7 Breakdown by Transport Mode Mix

# %%
mode_results = evaluator.summary_by_stratum(results_df, stratum_col="mode_mix")
print("📊 Results by Mode Mix:")
display(mode_results)

# %% [markdown]
# ### 4.8 Deadline Buffer Impact
#
# How does the deadline buffer affect the difference between planners?
# With tight deadlines (buffer=0), the robust planner should deviate more
# from the standard planner.

# %%
buffer_results = evaluator.summary_by_stratum(results_df, stratum_col="deadline_buffer_sec")
print("📊 Results by Deadline Buffer:")
display(buffer_results)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Success rate by buffer
ax = axes[0]
for q in Q_LEVELS:
    q_data = buffer_results[buffer_results['confidence_q'] == q]
    ax.plot(q_data['deadline_buffer_sec'], q_data['rob_success_rate'],
            'o-', label=f'Robust Q={q}', markersize=8)
std_data = buffer_results[buffer_results['confidence_q'] == Q_LEVELS[0]]
ax.plot(std_data['deadline_buffer_sec'], std_data['std_success_rate'],
        'ks--', label='Standard', markersize=10, linewidth=2)
ax.set_xlabel('Deadline Buffer (sec)')
ax.set_ylabel('Success Rate')
ax.set_title('Success Rate by Deadline Buffer')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Route identity by buffer
ax = axes[1]
for q in Q_LEVELS:
    q_data = buffer_results[buffer_results['confidence_q'] == q]
    ax.plot(q_data['deadline_buffer_sec'], q_data['routes_identical_pct'],
            'o-', label=f'Q={q}', markersize=8)
ax.set_xlabel('Deadline Buffer (sec)')
ax.set_ylabel('% Identical Routes')
ax.set_title('Route Identity by Deadline Buffer')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figs/e2e_buffer_impact.png', dpi=150)
plt.show()

# %% [markdown]
# ### 4.9 Timing Performance

# %%
print("📊 Average Planning Time:")
print(f"   Standard planner: {results_df['std_time_ms'].mean():.1f} ms")
print(f"   Robust planner:   {results_df['rob_time_ms'].mean():.1f} ms")
print(f"   Overhead:         {results_df['rob_time_ms'].mean() - results_df['std_time_ms'].mean():.1f} ms")

# %% [markdown]
# ## Phase 5: Summary Report

# %%
print("=" * 80)
print("  END-TO-END EVALUATION SUMMARY")
print("=" * 80)

# Count how many queries have different routes
q90_results = results_df[results_df['confidence_q'] == 0.90]
n_different = (~q90_results['routes_identical']).sum()
n_total = len(q90_results)
pct_different = n_different / n_total * 100 if n_total > 0 else 0

print(f"\n📊 At Q=0.90:")
print(f"   Total queries:              {n_total}")
print(f"   Routes differ:              {n_different} ({pct_different:.1f}%)")
print(f"   Standard success rate:      {q90_results['std_simulated_success'].dropna().mean():.3f}")
print(f"   Robust success rate:        {q90_results['rob_simulated_success'].dropna().mean():.3f}")
print(f"   Robust proposes earlier:    {q90_results['robust_proposes_earlier'].sum()} times")
print(f"   Robust filters routes:      {q90_results['robust_filters_routes'].sum()} times")

# Failed journeys analysis
failed_gt = q90_results[q90_results['journey_completed_gt'] == False]
if len(failed_gt) > 0:
    print(f"\n📊 For historically-failed journeys (ground truth):")
    print(f"   N queries:                  {len(failed_gt)}")
    print(f"   Standard still proposes:    {failed_gt['std_found_route'].sum()}")
    print(f"   Robust still proposes:      {failed_gt['rob_found_route'].sum()}")
    print(f"   Standard success rate:      {failed_gt['std_simulated_success'].dropna().mean():.3f}")
    print(f"   Robust success rate:        {failed_gt['rob_simulated_success'].dropna().mean():.3f}")

# Tight transfer analysis
tight = q90_results[q90_results['transfer_tightness'] == 'tight']
if len(tight) > 0:
    print(f"\n📊 For tight-transfer queries (< 5 min headway):")
    print(f"   N queries:                  {len(tight)}")
    print(f"   Standard success rate:      {tight['std_simulated_success'].dropna().mean():.3f}")
    print(f"   Robust success rate:        {tight['rob_simulated_success'].dropna().mean():.3f}")
    print(f"   Routes differ:              {(~tight['routes_identical']).sum()}")

print("\n" + "=" * 80)

# %%

# %%
