# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
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
# ---
# # DSLab Assignment 2 - Uncovering Public Transport Conditions using SBB data
#
# In this notebook, we use temporal information on SBB train operations to analyze delay patterns and develop delay prediction models.
#
# The workflow is structured as follows:
#
# - Data exploration and understanding, including data quality assessment
# - Integration of SBB data from multiple sources and reconciliation of identifiers
# - Definition and validation of delay measures
# - Analysis of when and where delays occur, including their temporal and spatial distribution
# - Development of a delay prediction model
#
# Throughout the notebook, you will be guided by a series of questions that structure your investigation. Exact numerical answers are not expected; the emphasis is on methodology, analytical reasoning, and how you interpret and use the data. Approach this as a real-world data science problem rather than an exercise with a single correct solution. We encourage an iterative exploration.
#
# ## Hand-in Instructions:
#
# - __Due: 05.05.2026~ 23:59:59 CET__
# - your gitlab project must be private, under your gitlab group https://dslabgit.datascience.ch/students/2026/{group_id}/assignment-2
# - add necessary comments and discussion to make your codes readable
# - make sure that your code is runnable

# %% [markdown]
# ---
# <div style="font-size: 100%" class="alert alert-block alert-info">
#     <b>ℹ️  Fair Cluster Usage:</b> As there are many of you working with the cluster, we encourage you to:
#     <ul>
#         <li>Whenever possible, prototype your queries on small data samples or partitions before running them on whole datasets</li>
#         <li>Save intermediate data in your HDFS home folder <b>f"/user/{username}/..."</b> or <b>f"/user/groups/com-490/{groupName}/..."</b></li>
#         <li>Convert the data to an efficient storage format when this is an option</li>
#         <li>Use spark <em>cache()</em> and <em>persist()</em> methods wisely to reuse intermediate results</li>
#     </ul>
# </div>

# %% [markdown]
# ---
# ## Start a spark Session environment

# %% [markdown]
# We provide the `username` and `hadoopFS` as Python variables accessible in both environments. You can use them to enhance the portability of your code, as demonstrated in the following Spark SQL command. Additionally, it's worth noting that you can execute Iceberg SQL commands directly from Spark on the Iceberg data.
#
# You must set the `groupName` variable to your group name.
#

# %%
import os
import pwd
import numpy as np
import pandas as pd
import sys

from pyspark.sql import SparkSession
from random import randrange
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import DecisionTreeRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.sql.window import Window
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

username = pwd.getpwuid(os.getuid()).pw_name
hadoopFS=os.getenv('HADOOP_FS', None)
groupName = "J1"

print(os.getenv('SPARK_HOME'))
print(f"hadoopFSs={hadoopFS}")
print(f"username={username}")
print(f"group={groupName}")

# %%
spark = (SparkSession\
            .builder
            .appName(username + '-assignment-2')
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

# %%
spark.sparkContext

# %%

# %% [markdown]
# Be nice to others - remember to add a cell `spark.stop()` at the end of your notebook.

# %% [markdown]
# ---
# ## Working Environment and Data Access
#
# This section introduces the working environment and key tools needed to explore and analyze the data. It covers practical methods for using Spark, managing intermediate results, and accessing SBB datasets, providing the foundation for the exercises that follow.
#
# ### How to approach the exercises
#
# You are expected to use Spark to solve the querstions of this assignment.
#
# You are allowed to use Spark SQL queries and PySpark DataFrame or RDD APIs.
#
# For instance, both are allowed:
#
# ```python
# spark.sql("""SELECT COUNT(DISTINCT stop_id) AS n_stop FROM iceberg.sbb.stops WHERE pub_date >= '2026-01-01'""").show()
# ```
#
# or:
#
# ```python
#  (spark.table("iceberg.sbb.stops")
#     .filter(F.col("pub_date") >= "2026-01-01")
#     .agg(F.countDistinct("stop_id").alias("n_stop"))
#  ).show()
# ```

# %% [markdown]
# ### User-defined and built-in functions
#
# In Spark Dataframes API you can create your own user defined functions for your Spark queries.
#
# You can use the _pyspark.sql.functions.udf_ method or _pyspark.sql.functions.pandas_udf_ as follows
#
# ```python
# @F.udf
# def lowercase(text):
#     """Convert text to lowercase"""
#     return text.lower()
# ```
#
# Or
# ```python
# @F.pandas_udf("string")
# def lowercase(text: pd.Series) -> pd.Series:
#     """Convert text to lowercase"""
#     return text.str.lower()
# ```
#
# Between the two, prefer the pandas udf, because the non-pandas move data between JVM and Python and are slower.
#
# So, for example, if we wanted to make a user-defined python function that returns a string value in lowercase, we could do something like this:

# %% [raw]
# @F.pandas_udf("string")
# def lowercase(text: pd.Series) -> pd.Series:
#     """Convert text to lowercase"""
#     return text.str.lower()

# %% [markdown]
# The _@F.udf_ and _@F.pandas_udf_ decorators turn Python functions into UDF objects that can be used in the DataFrame API -- and in this case are equivalent to:
#
# ```python
# def lowercase(text):
#     return text.lower()
#     
# lowercase = F.udf(lowercase)
# ```
#
# It basically takes our function and adds to its functionality. In this case, it registers our function as a pyspark dataframe user-defined function (UDF).
#
# Using these UDFs is very straightforward and analogous to other Spark dataframe operations. For example:

# %% [raw]
# %%time
# (
#     spark.table("iceberg.sbb.istdaten")
#         .select(F.col("operator_abrv"),lowercase(F.col("operator_abrv").alias('lowercase_op')
# ))).show(n=5)

# %% [markdown]
# To make a Python UDF (regular or pandas) available in SQL queries, you must register it with a **global** name in the Spark session using _spark.udf.register_. This gives SQL access to the function.

# %% [raw]
# %%time
# def uppercase(x):
#     x.str.upper()
# spark.udf.register("lowercase", lowercase)
# spark.sql("SELECT operator_abrv, lowercase(operator_abrv) as lowercase_op FROM iceberg.sbb.istdaten LIMIT 5").show(5)

# %% [markdown]
# The DataFrame API includes many other built-in functions, including the function for converting strings to lowercase.
# Other handy built-in dataframe functions include functions for transforming date and time fields.
#
# Note that the functions can be combined. Consider the following dataframe and its transformation:
#
# ```python
# from pyspark.sql import Row
#
# # create a sample dataframe with one column "degrees" going from 0 to 180
# test_df = spark.createDataFrame(spark.sparkContext.range(180).map(lambda x: Row(degrees=x)), ['degrees'])
#
# # define a function "sin_rad" that first converts degrees to radians and then takes the sine using built-in functions
# sin_rad = F.sin(F.radians(test_df.degrees))
#
# # show the result
# test_df.select(sin_rad).show()
# ```

# %% [markdown]
# ### DataFrame Aggregations
#
# The _Spark.DataFrame.groupBy_ does not return another DataFrame, but a _GroupedData_ object instead. This object extends the DataFrame with methods that allow you to do various transformations and aggregations on the data in each group of rows. 
#
# Conceptually the procedure is a lot like this:
#
# ![groupby](./figs/sgCn1.jpg)
#
#
# The column set used for the _groupBy_ is the _key_ - and it can be a list of column keys, such as _groupby('key1','key2',...)_ - all rows in a _GroupedData_ have the same key, and various aggregation functions can be applied on them to generate a transformed DataFrame. In the above example, the aggregation function is a simple `sum`.
#
#
# Refs:
# - [Dataframe API](https://spark.apache.org/docs/3.5.7/api/python/reference/pyspark.sql/dataframe.html)
# - [GroupedData API](https://spark.apache.org/docs/3.5.7/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.html)

# %% [raw]
# %%time
# (
#     spark.table("iceberg.sbb.istdaten")
#     .groupBy("operator_abrv")
#     .agg(F.count("*").alias("count"))
#     .orderBy(F.col("count").desc())
# ).show()

# %% [raw]
# %%time
# spark.sql("SELECT operator_abrv, COUNT(*) AS count FROM iceberg.sbb.istdaten GROUP BY operator_abrv ORDER BY count DESC").show()

# %% [markdown]
# ### Managing intermediate results
#
# Spark queries can be resource-intensive and time-consuming. We recommend first experimenting with smaller samples of the datasets. Once you are satisfied with the results, save the intermediate outputs in your HDFS user or group folder. This way, you can avoid repeatedly rerunning the most expensive queries.
#
# For instance to save a sample of table files.
#
# ```python
#     # Read a subset of the a dataset into a spark DataFrame (could also be result of spark.sql, etc).
#     df_sample = spark.read.csv(f'/data/com-490/bronze/sbb/istdaten/year=2025', sep=';', header=True).sample(0.01)
#     df_sample.filter(df_sample.BPUIC.startswith('85')).show(5)
#     
#     # Save DataFrame sample, under /user/groups/com-490/{groupName}, or /user/{username}, e.g.:
#     df_sample.write.parquet(f'/user/groups/com-490/{groupName}/assignment/istdaten_sample.parquet', mode='overwrite')
#
#     # ...
#     df_sample = spark.read.parquet(f'/user/groups/com-490/{groupName}/assignmen/istdaten_sample.parquet')
#
#     # Save under /user/{username}/assignment-3/warehouse/{username}.db/mytable (from the spark_catalog config)
#     df_sample.write.format("parquet").mode("overwrite").saveAsTable("mytable")
#     spark.sql("SELECT * FROM mytable LIMIT 5") # Note: drop table mytable will erase the data!
#     # ...
# ```

# %% [markdown]
# And to create a temp view:

# %% [raw]
# %%time
# spark.read.options(header=True).csv(f'/data/com-490/bronze/weather/stations').withColumns({
#       'lat': F.col('lat').cast('double'),
#       'lon': F.col('lon').cast('double'),
#     }).createOrReplaceTempView("weather_stations")

# %% [raw]
# spark.sql(f'SELECT * FROM weather_stations').printSchema()

# %% [raw]
# %%time
# spark.sql(f'SHOW TABLES').show(truncate=False)

# %% [raw]
# %%time
# spark.table("weather_stations").show(5)

# %% [raw]
# spark.sql(f'SELECT * FROM weather_stations LIMIT 5').toPandas()

# %% [raw]
# spark.sql(f'DROP VIEW weather_stations')

# %% [markdown]
# 💡 Notes:
# - Because Spark writes data in parallel across partitions, the original row order may not be preserved when saving to files. To preserve order, you can use _coalesce(1)_ to reduce the output to a single file. This is the opposite of _repartition(n)_, but keep in mind that writing everything to one file can hurt performance for larger datasets.
# - Do not hesitate to create temporary views out of tabular, data stored on file or from SQL queries. That will make your code reusable, and easier to read
# - You can convert spark DataFrames to pandas DataFrames inside the spark driver to process the results. But only do this for small result sets, otherwise your spark driver will run OOM.
#
# For instance:

# %% [markdown]
# ### Overview of SBB datasets
#
# For your convenience, the Spark session is preconfigured with a default _spark_catalog_ and our _iceberg_ catalog containing the SBB data.
#
# Run the code below to create your schema _spark_catalog.{username}_ and set it as your default; this is where your tables and views will be stored.
#
# Next, check the Iceberg catalog to see the available SBB tables.
#
# Pay attention to the full table names (_catalog.namespace.table_), as you will need them in later exercises.
#
# Ref:
#
# - [SBB Open data](https://opendata.swiss/en/organization/schweizerische-bundesbahnen-sbb)
# - [Open transport data](https://opentransportdata.swiss/en/)

# %%
# %%time
spark.sql(f'CREATE SCHEMA IF NOT EXISTS spark_catalog.{username}')

# %%
# %%time
spark.sql(f'USE spark_catalog.{username}')

# %%
# %%time
spark.sql(f'SHOW CATALOGS').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW SCHEMAS IN spark_catalog').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW TABLES IN spark_catalog.{username}').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW SCHEMAS IN iceberg').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW TABLES IN iceberg.sbb').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW TABLES IN iceberg.geo').show(truncate=False)

# %%
# %%time
spark.table("iceberg.sbb.stop_times").printSchema()

# %%
# %%time
spark.table("iceberg.sbb.trips").printSchema()

# %%
# %%time
spark.table("iceberg.sbb.routes").printSchema()

# %%
# %%time
spark.table("iceberg.sbb.calendar").printSchema()

# %%
# %%time
spark.table("iceberg.sbb.calendar_dates").printSchema()

# %%
# %%time
spark.table("iceberg.sbb.istdaten").printSchema()

# %%
# %%time
spark.table("iceberg.geo.shapes").printSchema()

# %% [markdown]
# ## Part I: First Steps with Spark DataFrames using SBB Data (15 points)
#
# Data from the SBB timetables and the Istdaten originate from different sources. One of the main challenges in integrating such data is reconciling object identifiers, such as those for stops, trips, and providers, and accurately matching corresponding entities across the various datasets.
#
# A useful first step is to perform basic statistical measurements to estimate the level of potential correspondence between datasets. For example, counting how many stops share the same identifier in both sources. While this does not guarantee that the identifiers refer to the same real-world entities, it provides an initial sense of scale and helps determine whether attempting a detailed matching process is worthwhile.

# %% [markdown]
# ### I.a Count distinct SBB stops and trips in official SBB time tables 2/15
#
# Several timetables are available in the SBB trips tables.
#
# For the year 2026, use Spark to compute, for each publication date, the approximate number of distinct trips, and distinct stops.
#
# Consider only stops in Switzerland (7-digit codes starting with 85, excluding platform information from the code).
#
# Count the number of stops, and number of distinct trips that serve at least one of these stops. 
#
# Make sure your query is as efficient as possible.
#
# ```bash
# root
#  |-- pub_date: date (nullable = true)
#  |-- n_trip: long (nullable = false)
#  |-- n_stop: long (nullable = false)
# ```
#
# E.g.:
#
# ```
# +----------+-------+------+
# |  pub_date| n_trip|n_stop|
# +----------+-------+------+
# |2026-01-**|*******| *****|
# |2026-01-**|*******| *****|
# ```
#
#

# %%
# %%time
sbb_trips_stops_df = (
    spark.table("iceberg.sbb.stop_times")
    # Restrict to 2026 Swiss stops and remove platform-level information.
    .filter(
        F.col("pub_date").between("2026-01-01", "2026-12-31") & 
        F.col("stop_id").startswith("85")
    )
    .groupBy("pub_date")
    .agg(
        F.approx_count_distinct("trip_id").alias("n_trip"),
        F.approx_count_distinct(F.substring("stop_id", 1, 7)).alias("n_stop"),
    )
)
(sbb_trips_stops_df.orderBy("pub_date")
).show()

# %% [markdown]
# ### I.b Compare results of I.a with actual Istdaten Data 2/15
#
# Compare these results with the number of distinct stops in Switzerland, and the distinct trips serving these stops, all from the istdaten table, computed over overlapping 7-day periods starting from each publication date.
#
# For instance: if the publication dates are 2026-01-02, and 2026-01-05, then count the number of distinct trips, and number of disticnt stops in the intervals 2026‑01‑02 to 2026‑01‑08 (included), then in the interval 2026‑01‑05 to 2026‑01‑11, and so on. Do not count _unplanned_ trips.
#
# Do you notice any systematic difference between the counts in the official timetables and the counts in the istdaten measurements?
#
# - Are the overall orders of magnitude similar?
# - How would you explain these differences?
# - What do these differences imply about the feasibility of matching istdaten stops to timetable stops, and aligning trips between the two datasets?
#
# ```bash
# root
#  |-- pub_date: date (nullable = true)
#  |-- n_trip: long (nullable = false)
#  |-- n_stop: long (nullable = false) 
# ```
#
# E.g.:
#
# ```
# +----------+------+------+
# |  pub_date|n_trip|n_stop|
# +----------+------+------+
# |2026-01-**|******| *****|
# |2026-01-**|******| *****|
# ```
#

# %%
# %%time
pub_dates_df = sbb_trips_stops_df.select("pub_date").distinct()

istdaten_filtered = (
    spark.table("iceberg.sbb.istdaten")
    .filter(F.col("operating_day").between("2026-01-01", "2026-12-31"))
    .filter(F.col("unplanned") == False)
    .filter(F.col("bpuic").cast("string").startswith("85"))
)

sbb_istdaten_trips_stops_df = (
    pub_dates_df.join(
        istdaten_filtered,
        F.col("operating_day").between(F.col("pub_date"), F.date_add(F.col("pub_date"), 6))
    )
    .groupBy("pub_date")
    .agg(
        F.approx_count_distinct("trip_id").alias("n_trip"),
        F.approx_count_distinct(F.substring(F.col("bpuic").cast("string"), 1, 7)).alias("n_stop")
    )
)

(sbb_istdaten_trips_stops_df.orderBy("pub_date")).show()

# %% [markdown]
# #### Are the overall orders of magnitude similar ?
# - For **Stops**: One can argue that they are close enough since both are computed based on approximations, but in my opinion No they are not close.
# - For **Trips**: No, there is a massive systematic difference. The timetable lists ~1.3 million trips per publication, while the 7-day Istdaten window only captures about ~330,000 trips.

# %%
# %%time
validity_check_df = (
    spark.table("iceberg.sbb.calendar")
    .filter(F.col("pub_date").between("2026-01-01", "2026-01-31"))
    .groupBy("pub_date")
    .agg(
        F.min("start_date").alias("earliest_planned_date"),
        F.max("end_date").alias("latest_planned_date"),
        # Calculate how many days this timetable spans
        F.datediff(F.max("end_date"), F.min("start_date")).alias("span_in_days")
    )
    .orderBy("pub_date")
)

validity_check_df.show()

# %% [markdown]
# #### How would you explain these differences?
# - For **Stops**:
#   1. Hardware limitations: Many small, rural stops or older vehicles lack the GPS/network hardware to report real-time arrival pings.
#   2. [REFUTED] On-Demand Stops: Stops marked as "Halt auf Verlangen" [stop on request](https://www.sbb.ch/en/offers/on-demand-services-taxi) will not generate an actual arrival record if no passengers requested the stop during our 7-day window.
# - For **Trips**:
#     publication's timetable contains the schedule for an entire year, whereas the Istdaten query only looks at a **7-day operational window**. We confirmed this assumption using the iceberg.sbb.calendar table (see code cell above), which proves that the scheduled trips for these publication dates span a 363-day validity period.

# %% [markdown]
# #### What do these differences imply about the feasibility of matching istdaten stops to timetable stops, and aligning trips between the two datasets?
# - Feasibility for Future Modeling: Because the vast majority of stops do overlap, aligning the datasets for delay prediction is highly feasible. However, this proves we must handle missing data carefully in our machine learning pipeline if we try to predict delays for every single planned stop, we will run into null values for the stations lacking tracking hardware.

# %% [markdown]
# ### I.c How many stops are both the Sbb timetables and Istdaten 3/15
#
# For each publication date in 2026, compute the number of stops that appear in both the SBB timetables and the 7-day Istdaten window starting from the same publication date.
#
# ```bash
# root
#  |-- pub_date: date (nullable = true)
#  |-- n_stop: long (nullable = false) 
# ```
#
# E.g.:
#
# ```
# +----------+------+
# |  pub_date|n_stop|
# +----------+------|
# |2026-01-**| *****|
# |2026-01-**| *****|
# ```

# %%
# %%time
timetable_stops = (
    spark.table("iceberg.sbb.stop_times")
    .filter(F.col("pub_date").between("2026-01-01", "2026-12-31"))
    .filter(F.col("stop_id").startswith("85"))
    .select(
        "pub_date",
        F.substring("stop_id", 1, 7).alias("base_stop_id")
    )
    .distinct() 
)

pub_dates_df = timetable_stops.select("pub_date").distinct()
istdaten_stops = (
    spark.table("iceberg.sbb.istdaten")
    .filter(F.col("operating_day").between("2026-01-01", "2026-12-31"))
    .filter(F.col("bpuic").between(8500000, 8599999))
    .select(
        "operating_day",
        F.substring(F.col("bpuic").cast("string"), 1, 7).alias("base_stop_id")
    )
    .distinct()
)
istdaten_window_stops = (
    pub_dates_df.join(
        istdaten_stops,
        F.col("operating_day").between(F.col("pub_date"), F.date_add(F.col("pub_date"), 6))
    )
    .select("pub_date", "base_stop_id")
    .distinct()
)

sbb_intersect_stops_df = (
    timetable_stops.join(
        istdaten_window_stops, 
        on=["pub_date", "base_stop_id"], 
        how="inner"
    )
    .groupBy("pub_date")
    .agg(F.count("*").alias("n_stop"))
)

(sbb_intersect_stops_df.orderBy("pub_date")).show()

# %%
# # %%time
raw_untracked_stops_df = (
    timetable_stops.join(
        istdaten_window_stops,
        on=["pub_date", "base_stop_id"],
        how="left_anti"  
    )
)

# Isolate the missing stops for just one publication date to test the theory
target_date = "2026-01-03"
missing_stops_single_day = raw_untracked_stops_df.filter(F.col("pub_date") == target_date)

# Get the stop_times details for these missing stops
missing_stop_details = (
    spark.table("iceberg.sbb.stop_times")
    .filter(F.col("pub_date") == target_date)
    .withColumn("base_stop_id", F.substring("stop_id", 1, 7))
    .join(missing_stops_single_day, on=["pub_date", "base_stop_id"], how="inner")
)

# Bring in the Trips and Routes tables so we know WHAT KIND of vehicle this was
trips_df = spark.table("iceberg.sbb.trips").filter(F.col("pub_date") == target_date)
routes_df = spark.table("iceberg.sbb.routes").filter(F.col("pub_date") == target_date)

missing_analysis_df = (
    missing_stop_details
    .join(trips_df, on=["pub_date", "trip_id"], how="left")
    .join(routes_df, on=["pub_date", "route_id"], how="left")
)

proof_df = (
    missing_analysis_df
    .groupBy("route_desc")
    .agg(
        F.countDistinct("base_stop_id").alias("n_missing_stops"),
        
        # Count how many of these scheduled stops were explicitly flagged as "On-Demand"
        F.sum(
            F.when((F.col("pickup_type") == 3) | (F.col("drop_off_type") == 3), 1).otherwise(0)
        ).alias("on_demand_occurrences")
    )
    .orderBy(F.desc("n_missing_stops"))
)

proof_df.show(20, truncate=False)

# %% [markdown]
# The vast majority of missing stops belong to Buses (B), Cable Cars (PB, GB), and Boats (BAT). This perfectly aligns with our hypothesis: these are smaller, rural, or non-train vehicles that likely lack the hardware for real-time pinging

# %% [markdown]
# ### I.d Explore route types and route descriptions 2/15
#
# List all distinct (_route_desc_, _route_type_) pairs from the SBB routes table published since the beginning of 2023.
#
# ```bash
# root
#  |-- route_desc: string (nullable = true)
#  |-- route_type: integer (nullable = true)
# ```
#
# E.g.:
#
# ```
# +----------+----------+
# |route_desc|route_type|
# +----------+----------+
# |       ABC|       104|
# |       DEF|      1700|
# ```
#
# Can you infer the mode of transport (rail, road, water, funicular/special rail or lift) from the _route_type_ codes? For instance what type of transport are codes in the 700 range (700, 705, 710, ...)
#

# %%
# %%time
sbb_route_types_df = (
    spark.table("iceberg.sbb.routes")
    .filter(F.col("pub_date") >= "2023-01-01")
    .select("route_desc", "route_type")
    .distinct()
)
(sbb_route_types_df.orderBy("route_type")).show(50)

# %% [markdown]
# #### Can you infer the mode of transport (rail, road, water, funicular/special rail or lift) from the route_type codes? For instance what type of transport are codes in the 700 range (700, 705, 710, ...) ?
# [REFERENCE](https://data.opentransportdata.swiss/en/dataset/vm-liste) in transportsubmodes.csv
#
# 1. Range: 100 – 117 --> **Heavy Rail / Trains** IC, IR, R, S, TGV, EC, PE
# 2. Range: 700 – 715 (and 202) --> Road (Bus) B (Bus), BN (Night Bus), BP (PostBus), EV (Replacement Bus).
# 3. Range: 1000 -->  Water (Ships & Ferries) BAT (Bateau/Boat), FAE (Ferry)
# 4. Range: 1300 – 1700 (1500 excluded) --> GB, PB, SL, FUN, CC, ASC
# 5. Metro (401) --> Metro as M
# 6. Tram (900) --> Tram as T
# 7. and more

# %% [markdown]
# ### I.e Compare results of I.c with SBB Istdaten 3/15
#
# Look at the list of _(transport, product_id)_ pairs from the SBB Istdaten table over the same period. Identify any inconsistencies or ambiguous cases, and propose a strategy for handling them; whether by correcting, ignoring, or otherwise disambiguating the entries.
#
# As a first step, compute the frequency of each _(transport, product_id)_ pair. This allows you to distinguish between rare, potentially invalid combinations and more frequent, likely valid ones, providing a data-driven basis for deciding how to treat ambiguous cases (Hint: each _transport_ should correspond to only one _product_id_.)
#
# ```bash
# root
#  |-- transport: string (nullable = true)
#  |-- product_id: string (nullable = true)
#  |-- frequency: long (nullable = false) 
# ```
#

# %%
# %%time
sbb_istdaten_transport_product_df = (
    spark.table("iceberg.sbb.istdaten")
    .filter(F.col("operating_day").between("2026-01-01", "2026-12-31"))
    .groupBy("transport", "product_id")
    .agg(F.count("*").alias("frequency"))
    .orderBy("transport", F.col("frequency").desc())
)

(sbb_istdaten_transport_product_df).show(50, truncate=False)


# %% [markdown]
# Provide a best-effort Spark application or user-defined function (using either Spark SQL or the DataFrame API) that cleans, corrects, and standardizes the values of _transport_ and _product_id_ in the Istdaten table. Then apply this transformation to the Istdaten dataset over the same time period and recompute the statistics over the corrected fields.
#
# The objective is to derive a more reliable feature from the _transport_ type and _product_id_ that you can use later in delay prediction, by resolving inconsistencies such as differing capitalizations, abbreviations(e.g., Bus vs B vs bus, etc.), or ignoring incorrect mappings.
#
# You may use Spark built-in functions such as _upper()_ or _lower()_, and you are encouraged to define custom mapping or correction functions where appropriate.
#
# Do not hesitate to leverage insights from the (route_desc, route_type) mapping in the SBB timetable to improve the consistency of the Istdaten mapping.
#
# Apply the method to the SBB Istdaten table  to return an improved list of _(transport,product_id)_ pairs.

# %% [markdown]
# ### Answers:
# #### A. Problem Identification:
# - **Casing issues**: Bus vs BUS.
#
# - **Typos/Errors**: B is mapped to Bua (typo) and Tram (completely wrong vehicle).
#
# - **Missing values**: IC, IR, S, etc., sometimes have an empty string "" for their product_id.
#
# - **Non-standard abbreviations**: Bus is used as a transport abbreviation instead of the standard B or BN.
#
# #### B. Strategy for Standardization :
#
# 1. Trim and Case Standardization: First, we strip any leading/trailing whitespaces and convert everything to UPPERCASE to immediately resolve "Bus" vs "BUS" vs "bus" mismatches.
#
# 2. Abbreviation Unification: We map non-standard transport abbreviations (like BUS) back to their official SBB timetable abbreviation (B).
#
# 3. Frequency-Driven Enforced Mapping: Using the frequency table from **I.e** and the route_desc codes from **I.d** , we force a 1-to-1 relationship. For example, if the transport is B, BN, EV, or CAR (which I.d showed are all in the **700s/200s range**), we explicitly force the product_id to be BUS. This instantly overwrites the errors like "Tram" or "Bua".

# %%
# %%time
def process_istdaten_stats(df):
    cleaned_istdaten_df = (
        df
        .withColumn("clean_transport", F.trim(F.upper(F.col("transport"))))
        .withColumn("clean_product_id", F.trim(F.upper(F.col("product_id"))))
        
        .withColumn("clean_transport", 
            F.when(F.col("clean_transport") == "BUS", "B")
            .otherwise(F.col("clean_transport"))
        )
        
        .withColumn("clean_product_id",
            # BUS CATEGORY (200s, 700s)
            F.when(F.col("clean_transport").isin("CAR", "EV", "KB", "B", "EXB", "BN", "BP", "RUB"), "BUS")
            # TRAM CATEGORY (900s)
            .when(F.col("clean_transport") == "T", "TRAM")
            # METRO CATEGORY (400s)
            .when(F.col("clean_transport") == "M", "METRO")
            # WATER CATEGORY (1000s)
            .when(F.col("clean_transport").isin("BAT", "FAE"), "SCHIFF")
            # ZAHNRADBAHN / CABLE CATEGORY (1300s, 1400s, 1700s)
            .when(F.col("clean_transport").isin("GB", "PB", "SL", "ASC", "FUN", "CC"), "ZAHNRADBAHN")
            # TRAIN CATEGORY (100s - Catch-all for heavy rail codes)
            .when(F.col("clean_transport").isin(
                "ZUG", "TGV", "EST", "IC", "ICE", "RJX", "EC", "IR", "IRE", "ATZ", 
                "ARZ", "NJ", "EN", "RB", "TER", "RE", "R", "PE", "S", "SN", "EXT"
            ), "ZUG")
            
            # If it's a completely empty transport code but has 'ZUG', keep 'ZUG'
            .when((F.col("clean_transport") == "") & (F.col("clean_product_id") == "ZUG"), "ZUG")
                    .otherwise(F.col("clean_product_id"))
        )
    )

    sbb_cleaned_stats_df = (
        cleaned_istdaten_df
        .groupBy("clean_transport", "clean_product_id")
        .agg(F.count("*").alias("frequency"))
        .filter(F.col("clean_transport") != "")
        .orderBy("clean_transport", F.col("frequency").desc())
    )
    
    return sbb_cleaned_stats_df


# %%
# %%time
istdaten_df = (
    spark.table("iceberg.sbb.istdaten")
    .filter(F.col("operating_day").between("2026-01-01", "2026-12-31"))
)

final_stats_df = process_istdaten_stats(istdaten_df)
final_stats_df.show(50, truncate=False)

# %% [markdown]
# ### I.f Distribution of Actual Arrival Time measurement methods 3/20
#
# Actual arrival and departure times at stops in the Istdaten dataset are recorded using several methods with varying levels of precision. Among these, _REAL_ represents the highest level of accuracy, followed by _GESCHATZT_ (estimated), _PROGNOSE_ (forecast), and _UNBEKANNT_ (unknown, which should be ignored).
#
# Assess how frequently each recording method is used in the available Istdaten dataset to measure the arrival status, relative to the total number of measurements per year across all methods, and analyze how this distribution has evolved year over year since 2023. Ignore unplanned trips and when the arrival status is NULL.
#
# Note that the empty string ("") and PROGNOSE (forecast) are treated as equivalent, according to [Istdaten cookbook](https://opentransportdata.swiss/en/cookbook/historic-and-statistics-cookbook/actual-data/) and should be counted together as one.
#
#
#
# ```
# root
#  |-- year: integer (nullable = true)
#  |-- arr_status: string (nullable = true)
#  |-- frequency: long (nullable = false)
#  |-- total: long (nullable = true)
# ```
#
# E.g.:
#
# ```
# +----+----------+---------+---------+
# |year|arr_status|frequency|    total|
# +----+----------+---------+---------+
# |2023|      REAL|123456789|987654321|
# |2023|  PROGNOSE| 12345678|987654321|
# ...
# ```

# %%
from pyspark.sql.window import Window
arr_status_df = (
    spark.table("iceberg.sbb.istdaten")
    .withColumn("year", F.year("operating_day"))
    .filter(F.col("year") >= 2023)
    .filter(F.col("unplanned") == False)
        .filter(F.col("arr_status").isNotNull())
    .filter(F.upper(F.trim(F.col("arr_status"))) != "UNBEKANNT")
    
    .withColumn("clean_arr_status", 
        F.when(F.trim(F.col("arr_status")) == "", "PROGNOSE")
        .otherwise(F.upper(F.trim(F.col("arr_status"))))
    )
)

freq_df = (
    arr_status_df
    .groupBy("year", "clean_arr_status")
    .agg(F.count("*").alias("frequency"))
    .withColumnRenamed("clean_arr_status", "arr_status")
)

window_spec = Window.partitionBy("year")

distribution_df = (
    freq_df
    .withColumn("total", F.sum("frequency").over(window_spec))
    .select("year", "arr_status", "frequency", "total")
    .orderBy(F.col("year").desc(), F.col("frequency").desc())
)

distribution_df.show(20, truncate=False)

# %% [markdown]
# ### Analysis:
#
# **Improving Accuracy**: The percentage of REAL measurements representing the "highest level of accuracy" 
# has climbed steadily from 71.9% to 86.7%. This suggests that SBB and its partners have significantly expanded their real-time tracking infrastructure over the last few years.
#
# **Total Records**: 2026 is an incomplete year so the total is way less than the previous years which make sense. 

# %% [markdown]
# ## Part II: SBB Delay distributions (15 points)

# %% [markdown]
# ### II.a Identify public transport operators of the greater Lausanne region 3/15
#
# Using the last available day of actual SBB Istdaten data and the most recent SBB time table data provided to you, find the codes of all operators serving at least one of the following stops and for which real time information is available:
#
# - _Morges_
# - _Morges, gare_
# - _Lausanne_
# - _Lausanne, gare_
# - _Renens VD_
# - _Renens VD, gare_
#
# You may assume that the Id of a stop is the same in the SBB Istdaten table and in the time table.

# %%
# %%time

max_pub_date = spark.table("iceberg.sbb.stop_times").select(F.max("pub_date")).collect()[0][0]
max_op_day = spark.table("iceberg.sbb.istdaten").select(F.max("operating_day")).collect()[0][0]

target_stops = ['Morges', 'Morges, gare', 'Lausanne', 'Lausanne, gare', 'Renens VD', 'Renens VD, gare']

target_bpuics_df = (
    spark.table("iceberg.sbb.stops")  
    .filter(F.col("pub_date") == max_pub_date)
    .filter(F.col("stop_name").isin(target_stops))
    .withColumn("bpuic", F.substring("stop_id", 1, 7).cast("integer"))
    .select("bpuic")
    .distinct()
)

# Find the operators in Istdaten on the last available day serving these stops with real-time information available
lmr_gares_df = (
    spark.table("iceberg.sbb.istdaten")
    .filter(F.col("operating_day") == max_op_day)
    .join(target_bpuics_df, on="bpuic", how="inner")
    # Ensuring real-time data is available (based on Part I.f)
    .filter(
        (F.col("arr_status") == "REAL") | 
        (F.col("arr_status") == "PROGNOSE") | 
        (F.trim(F.col("arr_status")) == "")
    )
    .select("operator_id")
    .distinct()
)

lmr_gares_df.createOrReplaceTempView("lmr_gares")
lmr_gares_df.show(100,truncate=False)

# %% [markdown]
# For verification, can you find the name of the operators corresponding to the codes identified earlier using the most recent publication of the _agency_ table.
#
# The table is available in CSV format under HDFS: _/data/com-490/bronze/sbb/agency_.
#
# You should find Transport public de la région Lausannoise (TL), transports de la région Morges-Bière-Cossonay (MBC), SBB, etc.
#
# Hint: the operator Id of the agency table does not include the country code '85:', like istdaten does.

# %%
# %%time
agency_df = spark.read.options(header=True).csv(f'/data/com-490/bronze/sbb/agency')
agency_df.createOrReplaceTempView("agency") # Give a name to it, for spark.sql commands.

# Strip "85:" from operator_ids to match the agency_id format
stripped_operators_df = lmr_gares_df.withColumn(
    "agency_id", 
    F.regexp_replace(F.col("operator_id"), "^85:", "")
)

agency_df_filtered = (
    agency_df
    .join(stripped_operators_df, on="agency_id", how="inner")
)

agency_df_filtered.show(10,truncate=False,vertical=True)

# %%
# %%time
# Verification, count how often each operator appears in the SBB tables in January 2026.

# Count the actual appearances in January 2026 using istdaten
jan_counts_df = (
    spark.table("iceberg.sbb.istdaten")
    .filter(F.col("operating_day").between("2026-01-01", "2026-01-31"))
    .join(lmr_gares_df, on="operator_id", how="inner")
    .groupBy("operator_id")
    .agg(F.count("*").alias("appearance_count"))
)

# Join with agency_df_filtered to bring in the human names
final_verification_df = (
    jan_counts_df
    .join(
        agency_df_filtered.select("operator_id", "agency_name").distinct(), 
        on="operator_id", 
        how="left"
    )
    .orderBy(F.desc("appearance_count"))
    .select("operator_id", "agency_name", "appearance_count")
)

final_verification_df.show(50, truncate=False)

# %% [markdown]
# From now on, consider only trip IDs associated with the operator IDs identified earlier, as well as the stops served by those trips.

# %% [markdown]
# ### II.b Top average arrival delay 4/15
#
# Among the operators identified earlier, find those with the highest monthly 90th percentile delays. For each operator, compute the 90th percentile over all delays recorded during the month, regardless of the locations served; 90% of that operator’s delays fall below this value. Also include the number of observations recorded for each operator during the month (frequency).
#
# Display the results: for each (year, month) since 20203, sort the operators by the 90th percentile of delay in descending order, and display only the entries where this percentile exceeds 3 minutes and 30 seconds.
#
# The schema of the resulting Spark DataFrame should include:
# ```
# root
#  |-- year: integer (nullable = true)
#  |-- operator_id: string (nullable = true)
#  |-- month: integer (nullable = true)
#  |-- frequency: integer (nullable = true)
#  |-- p90_arr_delay: interval day to second (nullable = true)
# ```
#
# E.g.:
#
# ```
# +----+-----+-----------+-----------------------------------+---------+
# |year|month|operator_id|p90_arr_delay                      |frequency|
# +----+-----+-----------+-----------------------------------+---------+
# |2023|1    |85:12      |INTERVAL '0 00:04:38' DAY TO SECOND|12345    |
# |2023|1    |85:34      |INTERVAL '0 00:04:31' DAY TO SECOND|2345     |
# |2023|2    |85:34      |INTERVAL '0 00:04:13' DAY TO SECOND|34567    |
# |2023|2    |85:12      |INTERVAL '0 00:03:35' DAY TO SECOND|456789   |
# ```

# %%
# %%time

# DataFrame with all the required filters and the delay calculation
istdaten_valid_arrivals = (
    spark.table("iceberg.sbb.istdaten")
    .withColumn("year", F.year("operating_day"))
    .withColumn("month", F.month("operating_day"))
    .filter(F.col("year") >= 2023)
    .filter(F.col("unplanned") == False)        # TODO: why exclude unplanned
    .filter(F.col("arr_status").isNotNull())
    .filter(F.upper(F.trim(F.col("arr_status"))) != "UNBEKANNT")
    .filter(F.col("arr_actual").isNotNull() & F.col("arr_time").isNotNull())
    .withColumn(
        "delay_sec", 
        F.col("arr_actual").cast("timestamp").cast("long") - F.col("arr_time").cast("timestamp").cast("long")
    )
)

# 90th percentile for the specific operators
op_avg_delay_df = (
    istdaten_valid_arrivals
    .join(lmr_gares_df, on="operator_id", how="inner")
    .groupBy("year", "month", "operator_id")
    .agg(
        F.expr("percentile_approx(delay_sec, 0.9)").alias("p90_sec"),
        F.count("*").alias("frequency")
    )
    # delays strictly greater than 3m30s (210 seconds) 
    .filter(F.col("p90_sec") > 210)
    .withColumn("p90_arr_delay", F.expr("INTERVAL '1' SECOND * p90_sec"))
    .select("year", "month", "operator_id", "p90_arr_delay", "frequency")
)

(op_avg_delay_df
     # Sort Year Ascending (oldest first)
     # Sort Month Ascending (Jan to Dec)
     # Sort Delay Descending (worst delays first)
    .orderBy(F.col("year").asc(), F.col("month").asc(), F.col("p90_arr_delay").desc())
).show(60,truncate=False)

# %% [markdown]
# Generate a Spark DataFrame containing only the stops served by Transport Public de la Région Lausannoise (TL) or Transports de la région Morges-Bière-Cossonay (MBC).
#
# Next, generate a Spark DataFrame to estimate the monthly 90th percentile of arrival delays for each stop identified earlier, considering all trips serving these stops, including those from other operators recorded in Istdaten since 2023. Also include the number of observations recorded for each stop during the month (frequency).
#
# Display the results: sort the table in descending order of the 90th percentile delay and retain only stops where it exceeds 10 minutes.
#
# _UNBEKANNT_ (unkown) measurement methods and unplanned trips should not be counted.
#
# The schema of the table is, at a minimum:
#
# ```
# root
#  |-- year: integer (nullable = true)
#  |-- month: integer (nullable = true)
#  |-- bpuic: integer (nullable = true)
#  |-- p90_arr_delay: interval day to second (nullable = true)
#  |-- frequency: integer (nullable = true)
# ```

# %%
# %%time

# From Part II.a
# 85:151 -> Transports Publics de la Région Lausannoise sa
# 85:29  -> Transports de la région Morges-Bière-Cossonay (MBC)
# 85:764 -> Automobiles MBC

# Distinct stops (bpuic) served by TL and MBC
tl_mbc_stops_df = (
    spark.table("iceberg.sbb.istdaten")
    .filter(F.col("operator_id").isin("85:151", "85:764", "85:29"))
    .select("bpuic")
    .distinct()
)

# 90th percentile arrival delay for these stops across ALL operators
stop_avg_delay_df = (
    istdaten_valid_arrivals
    .join(tl_mbc_stops_df, on="bpuic", how="inner") # Restrict to TL/MBC stops
    .groupBy("year", "month", "bpuic")
    .agg(
        F.expr("percentile_approx(delay_sec, 0.9)").alias("p90_sec"),
        F.count("*").alias("frequency")
    )
    # Exceeds 10 minutes (600 seconds)
    .filter(F.col("p90_sec") > 600)
    .withColumn("p90_arr_delay", F.expr("INTERVAL '1' SECOND * p90_sec"))
    .select("year", "month", "bpuic", "p90_arr_delay", "frequency")
)

(stop_avg_delay_df
     # Sort globally by the worst delays first
    .orderBy(F.col("p90_arr_delay").desc())
).show(60,truncate=False)

# %% [markdown]
# ### II.c Visualization 4/15
#
# Convert the Spark DataFrames of stop percentile delays calculated earlier to a Pandas DataFrame and **visualize the results** in the notebook.
#
# We are not looking for visual perfection, we just want to verify that your results are generally accurate (stop locations and delay hot spots). However, feel free to unleash your creativity and come up with a visualization that you find insightful. You are also encouraged to reuse and adapt visualization methods introduced in the lab exercises.
#
# 💡 Hints:
# - Do not hesitate to take advantage of _iceberg.sbb.stops_ if you would like to include geospatial information in your analysis, for instance to create a heatmap of stop delays. You may assume that all the SBB Istdaten _bpuic_ and the _stop_id_ from the most recent SBB timetables are consistent.
# - If using arrival delay heatmaps, you may want to cap the delays to 1h (i.e. delays above 1h are shown as 1h delay).

# %%
# %%time
import plotly.express as px

def displayDelays(spark_df):
    
    # Distinct stops and coordinates from the most recent timetable
    max_pub_date = spark.table("iceberg.sbb.stop_times").select(F.max("pub_date")).collect()[0][0]
    
    stops_df = (
        spark.table("iceberg.sbb.stops")
        .filter(F.col("pub_date") == max_pub_date)
        .withColumn("bpuic", F.substring("stop_id", 1, 7).cast("integer"))
        .select("bpuic", "stop_name", "stop_lat", "stop_lon")
        .distinct()
    )
    
    # Join the delays with the geospatial stops
    joined_df = spark_df.join(stops_df, on="bpuic", how="inner")

    
    # This forces Spark to finish the math before trying to stream the data to Pandas
    joined_df.cache().count()
    
    # Convert to a Pandas DataFrame
    pdf = joined_df.toPandas()
    pdf['delay_minutes'] = pdf['p90_arr_delay'].dt.total_seconds() / 60.0
    
    # Aggregate the delay (average 90th percentile across the months per stop)
    agg_pdf = pdf.groupby(['bpuic', 'stop_name', 'stop_lat', 'stop_lon'], as_index=False)['delay_minutes'].mean()
    
    # Cap the delays at 1 hour
    agg_pdf['delay_capped_min'] = agg_pdf['delay_minutes'].clip(upper=60.0)
    
    # Heatmap
    fig = px.density_mapbox(
        agg_pdf,
        lat='stop_lat',
        lon='stop_lon',
        z='delay_capped_min',
        hover_name='stop_name',
        hover_data={'delay_capped_min': ':.1f', 'stop_lat': False, 'stop_lon': False},
        radius=18, # Adjust this to make the heat blobs larger or smaller
        center=dict(lat=46.5197, lon=6.6323), # Centers the map on Lausanne
        zoom=10.5,
        mapbox_style="open-street-map",
        title="90th Percentile Arrival Delays Heatmap (Capped at 60 mins)",
        color_continuous_scale="Inferno",
        labels={'delay_capped_min': 'Avg 90th Pctl Delay (min)'},
        height=800,
        width=1000
    )
    fig.show()

displayDelays(
    stop_avg_delay_df
)

# %% [markdown]
# ### II.d Spark Partition Windows 4/15
#
# In a previous question, we calculated the monthly delay percentile at each stop.
#
# Now, suppose we want to determine, for each hour, a list of stops sorted from highest to lowest 90th arrival delay percentiles. This is a more complex task - it cannot be easily expressed with simple groupBy aggregations alone, as they collapse the data into a single row per group (including [time-based](https://spark.apache.org/docs/3.5.7/api/python/reference/pyspark.sql/api/pyspark.sql.functions.window.html) groupings). Instead, it requires a window function, which performs calculations across sets of rows related to the current row without collapsing them.
#
# A window function requires of a window specification, which defines how rows are partitioned, ordered within each partition, and optionally which frame of rows is visible to the calculation. The aggregation logic defines the computation applied to each window; rows are processed in the specified order, and the function produces a new column with the result for each row. You can apply multiple functions and even different windows per query.
#
#
# We recommend reading this [window functions article](./figs/introducing-window-functions-in-spark-sql.html)  and optionally the [Spark SQL](https://spark.apache.org/docs/3.5.7/sql-ref-syntax-qry-select-window.html) documentation to get acquainted with the idea.
#
# For each hour since 2023, compute the 90th-percentile arrival delay for each stop over the preceding hour. Assign ranks to the stops based on these percentiles, with rank 1 corresponding to the highest delay; stops with the same delay receive the same rank.
#
# Display: all stops up to rank 5 for each hour, ordered chronologically by hour and, within each hour, by ascending rank (i.e., highest delays first).
#
# Despite the complexity of the operation, it can be accomplished efficiently in just a few lines of python or SQL code!

# %% [markdown]
# Define a window function, that partitions the data by the actual arrival time columns rounded by hour. Within each partition, compute the hourly arrival time delay percentiles. Then, apply the _rank_ function over this window to assign rankings to the stops based on their respective delay percentiles within the hour. Finally, filter the results to keep only the top _N_ ranked (highest delay) stops within the window.
#
# 1. create a Spark DataFrame with the following columns: _bpuic_ (stop id) from Lausanne region identified earlier, actual arrival time (including date, since 2023) truncated to the hour, and arrival delay calculated from planned arrival time and actual arrival time.
#
# Minimum schema:
#
# ```
# root
#  |-- actual_arrival_hour: timestamp (nullable = true)
#  |-- bpuic: integer (nullable = true)
#  |-- arr_delay: interval day to second (nullable = true)
# ```
#
# 2. define a window (fixed-size, non-overlapping) to specify partitioning and ordering. Partition the data by _actual_arrival_hour_ and order rows within each partition by delay percentiles (e.g. _p_arr_delay_) in **descending** order (highest first).
#
# ```
# Window.partitionBy(...).orderBy(...) ...
# ```
#
# Or (SQL)
#
# ```
# SELECT ... window_function() OVER (PARTITION BY ... ORDER BY ... ) ...
# ```
#
# 3. Calculate the hourly arrival delay percentiles for each stop (either a groupby or a partitioning window would work here).
#
# 4. Define the ranking computation window.
#
# Use the helpful built-in F.rank() _spark.sql.function_, or SQL _RANK()_ to apply it over the hourly window. Give a name to the resulting column (alias), e.g. _rank_.
#
# 5. Apply the percentile computation on the DataFrame and the hourly rank window computation on the percentile computation.

# %%
# %%time

# tl_mbc_stops_df (Part II.b) to restrict the data to the Lausanne region
# istdaten_valid_arrivals (Part II.b) for the clean dat
# a
base_hourly_df = (
    istdaten_valid_arrivals
    .join(tl_mbc_stops_df, on="bpuic", how="inner")
    .withColumn("actual_arrival_hour", F.date_trunc("hour", F.col("arr_actual").cast("timestamp")))
    .withColumn("arr_delay", F.expr("INTERVAL '1' SECOND * delay_sec"))
    .select("actual_arrival_hour", "bpuic", "arr_delay", "delay_sec")
)


# %%
# %%time

from pyspark.sql.window import Window

hourly_rank_window = Window.partitionBy("actual_arrival_hour").orderBy(F.col("p_arr_delay").desc())

# %%
# %%time

rank_logic = F.rank().over(hourly_rank_window)

# %%
# %%time

final_ranked_stops_df = (
    base_hourly_df
    # Group by hour and stop to calculate the 90th percentile delay
    .groupBy("actual_arrival_hour", "bpuic")
    .agg(
        F.expr("percentile_approx(delay_sec, 0.9)").alias("p90_sec")
    )
    .withColumn("p_arr_delay", F.expr("INTERVAL '1' SECOND * p90_sec"))
    # Apply window function to calculate the rank of each stop within that hour
    .withColumn("rank", rank_logic)
    # Only the top 5 worst delays per hour and sorting
    .filter(F.col("rank") <= 5)
    .select("actual_arrival_hour", "bpuic", "p_arr_delay", "rank")
    .orderBy(F.col("actual_arrival_hour").asc(), F.col("rank").asc())
)

final_ranked_stops_df.show(50, truncate=False)

# %% [markdown]
# **Checkpoint:** The output should ressemble (actual values will differ):
#
# ```
# +-------------------+-------+-----------------------------------+----+
# |actual_arrival_hour|bpuic  |p_arr_delay                        |rank|
# +-------------------+-------+-----------------------------------+----+
# |        ....       |  ...  |                 ...               |....|
# |2023-01-01 03:00:00|8592010|INTERVAL '0 00:03:14' DAY TO SECOND|1   |
# |2023-01-01 03:00:00|8592141|INTERVAL '0 00:02:36' DAY TO SECOND|2   |
# |2023-01-01 03:00:00|8590442|INTERVAL '0 00:02:26' DAY TO SECOND|3   |
# |2023-01-01 03:00:00|8504176|INTERVAL '0 00:02:26' DAY TO SECOND|3   |
# |2023-01-01 03:00:00|8504177|INTERVAL '0 00:01:47' DAY TO SECOND|5   |
# (...)
# ```

# %% [markdown]
# ### II.e Overlapping Spark Windows - 2 (bonus)

# %% [markdown]
# In the previous question, we computed rankings over non-overlapping (tumbling) windows, where each window covers a fixed hourly interval and does not overlap with others.
#
# With window functions, you can also compute aggregates over a sliding window, where the window moves across the data and may overlap with previous intervals.
#
# **Follow-up question:**
#
# For each hour, compute the 90th-percentile arrival delay for each stop over the preceding 3-hour window. Assign ranks to the stops based on these percentiles, with rank 1 corresponding to the highest delay; stops with the same delay receive the same rank.
#
# Display all stops up to rank 5 for each hour, ordered chronologically by hour and, within each hour, by ascending rank.
#
# The process follows a similar pattern, with a few key differences:
#
# * Rows are processed independently for each stop.
# * The window is ordered by timestamp and, for each row, spans the preceding 3 hours up to the current timestamp (i.e., from t − 3h to t).
# * The percentile must be computed over all delay events within the window (it cannot be derived from pre-aggregated values, as percentiles are not composable.
#
#
# 💡 **Hints:**
# * [spark.sql.Window(Spec)](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/window.html)
# * Datetime times are not supported in sliding windows ranges, times must be converted to long

# %% [markdown]
# 1. Define a _pyspark.sql.window.WindowSpec_ to specify the sliding window partition, the row ordering inside the partition, and its 3h _range_.
#
# 2. Define a computation on the sliding window: calculate the delay percentile of of each stop. Use the helpful built-in F.approx_percentile() spark.sql.function.
#
# 4. Define a tumbling window partition to specify the tumbling 1h window partition.
#
# 5. Define a computation on the tumbling window: calculate the rank of the 
#
# 6. Apply 3 hour window to the DataFrame, and order chronologically.

# %%
sliding_base_df = (
    istdaten_valid_arrivals
    .join(tl_mbc_stops_df, on="bpuic", how="inner")
    .withColumn("actual_arrival_hour", F.date_trunc("hour", F.col("arr_actual").cast("timestamp")))
    .withColumn("hour_long", F.col("actual_arrival_hour").cast("long"))
)

# %%
# Sliding Window (3 Hours)
# 3 hours = 3*60*60 = 10,800 seconds
sliding_window = (
    Window.partitionBy("bpuic")
    .orderBy("hour_long")
    .rangeBetween(-10800, 0)
)

# 90th percentile of raw delay events over that 3h window
sliding_p90_logic = F.approx_percentile("delay_sec", 0.9).over(sliding_window)

# Apply the sliding window to our raw events
hourly_stop_3h_p90_df = (
    sliding_base_df
    .withColumn("p90_sec", sliding_p90_logic)
    .select("actual_arrival_hour", "bpuic", "p90_sec")
    .dropDuplicates(["actual_arrival_hour", "bpuic"])
    .withColumn("p_arr_delay", F.expr("INTERVAL '1' SECOND * p90_sec"))
)

# %%
# Tumbling Window (1 Hour)
# Partition by the hour to rank all stops against each other.
tumbling_window = Window.partitionBy("actual_arrival_hour").orderBy(F.col("p_arr_delay").desc())
rank_logic = F.rank().over(tumbling_window)

# %%
# Apply the rank, filter for the top 5, and order chronologically
final_sliding_ranked_stops_df = (
    hourly_stop_3h_p90_df
    .withColumn("rank", rank_logic)
    .filter(F.col("rank") <= 5)
    .select("actual_arrival_hour", "bpuic", "p_arr_delay", "rank")
    .orderBy(F.col("actual_arrival_hour").asc(), F.col("rank").asc())
)

final_sliding_ranked_stops_df.show(50, truncate=False)

# %% [markdown]
# #### Note on the Results: 
#
# By comparing this output to **Part II.d**, we can observe the fundamental differences between the two windowing techniques:
# * **The "Carryover" Effect:** In the sliding window, severe delays (like those at `04:00:00`) remain in the top 5 for the next three hours because the window reaches backward. The tumbling window resets entirely every hour.
# * **The "Dilution" Effect:** Stop `8501037` had a severe 17-minute delay in the 1-hour tumbling window at `05:00:00`. However, in the 3-hour sliding window, that spike is "diluted" by the inclusion of normal, on-time arrivals from the preceding two hours, dropping its 90th percentile significantly.
#

# %% [markdown]
# **Checkpoint:** The output should ressemble (actual values will differ):
#
# ```
# +-------------------+-------+-----------------------------------+----+
# |actual_arrival_hour|bpuic  |p_arr_delay                        |rank|
# +-------------------+-------+-----------------------------------+----+
# |        ...        |  ...  |               ...                 |....|
# |2023-01-01 03:00:00|8592010|INTERVAL '0 00:03:14' DAY TO SECOND|1   |
# |2023-01-01 03:00:00|8592141|INTERVAL '0 00:02:36' DAY TO SECOND|2   |
# |2023-01-01 03:00:00|8590442|INTERVAL '0 00:02:26' DAY TO SECOND|3   |
# |2023-01-01 03:00:00|8504176|INTERVAL '0 00:02:01' DAY TO SECOND|4   |
# |2023-01-01 03:00:00|8504177|INTERVAL '0 00:01:47' DAY TO SECOND|5   |
# |2023-01-01 04:00:00|8570179|INTERVAL '0 00:04:04' DAY TO SECOND|1   |
# (...)
# ```

# %% [markdown]
# ---
# ## PART III: SBB Data Aligment (10 points)

# %% [markdown]
# In this part, you will attempt to match trip IDs from the Istdaten dataset with trip IDs from the SBB timetable data.
#
# You should restrict the analysis to:
#
# - the last available week of SBB Istdaten data, and
# - the most recent publication of the SBB timetables.
#
# In addition, only consider:
#
# - trips operated by the operators identified in Part _IIa_, and
# - stops served by _TL_ and MBC, as identified in Part _IIb_.
#
# There are multiple ways to approach this problem in Spark, each with different trade-offs.
#
# This is an open-ended question with no single correct solution. Grading will depend on the soundness of your approach and on the number of trips successfully matched, as well as the confidence of these matches.
#
# 💡 Hints:
#
# - You may assume that operator and BPUIC IDs are reliable and consistent across the Istdaten and SBB timetable datasets.
# - Many SBB timetable trips do not report actual arrival times (e.g. M2).
# - Some timetable trips are redundant: the same line may depart from the same stop ID at the same time on the same day of the week (e.g. M1). A match is considered valid if an Istdaten trip ID corresponds to any of these redundant timetable entries

# %% [markdown]
# <hr>
#
# The general plan for matching will be to attempt to link istdaten and timetable trips with the same arrival/departure times and days, that belong to the same operator and sequence of stops.

# %% [markdown]
# Firstly, we'd like to restrict our SBB timetables and SBB istdaten data dataframes to trips operated by the operators identified in Part IIa and stops served by TLC and MBC, on the last week available week of istdaten data and the most recent publications of timetables.
#
# The timestamps were found earlier. The most recent timetable publication date is found under max_pub_date. The last available day of istdaten is stored under max_op_day, we then just need to filter between the range from max_op_day - 6 to max_op_day.
#
# The operators identified in Part IIa are stored in lmr_gares_df, thus we just need to keep trips with the operator_ids found in it. For istdaten data, this is easy enough, as we already have the operator_id column, and thus only need an inner join operation. For the timetable data, from the istdaten data documentation, we know that the operator_id is of the form CC:agency_id, with CC as the country code and agency_id a key in sbb.routes. Therefore, we just need to join on sbb.trips and then sbb.routes, joining on the specified agency_ids in the latter.
#
# The BPUICs of the stops served by TL and MBC are stored in tl_mbc_stops_df. Thankfully, istdaten already has BPUICs, so again, only a simple inner join is required. For the timetable data, we know that stop_ids are of the form CC:BPUIC, thus, we just need to split on the stop_id column and inner join.

# %%
from pyspark import StorageLevel

# %%
# %%time

"""
Filter timetable dataset to only contain trips operated by operators identified in Part IIa, contained in lmr_gares_df,
and stops served by TL and MBC, contained in tl_mbc_stops_df
"""

lmr_agency_ids = [row.agency_id for row in lmr_gares_df.withColumn("agency_id", F.split(F.col("operator_id"), ":")[1]).collect()]

stop_times_recent = (
    spark.table("iceberg.sbb.stop_times")
    # Latest publishing date
    .filter(F.col("pub_date") == max_pub_date)
    .select("trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence", "pub_date")
    .withColumn("bpuic", F.split(F.col("stop_id"), ":").getItem(0))
    .join(F.broadcast(tl_mbc_stops_df),
          on="bpuic", how="inner")
    .join(spark.table("iceberg.sbb.trips").filter(F.col("pub_date") == max_pub_date).select("route_id", "trip_id", "service_id"),
          on="trip_id", how="inner")
    .join(F.broadcast(spark.table("iceberg.sbb.routes").select("agency_id", "route_id", "route_desc").filter(F.col("agency_id").isin(lmr_agency_ids))), 
          on="route_id", how="inner")
)

stop_times_recent = stop_times_recent.persist(StorageLevel.MEMORY_AND_DISK)
print(stop_times_recent.count())

# %%
# %%time

"""
Filter istdaten dataset to only contain trips operated by operators identified in Part IIa and stops served by TL and MBC
"""

# Column object required in operation
max_op_date = F.lit(max_op_day)

istdaten_filtered = (
    spark.table("iceberg.sbb.istdaten")
    # Within the last week
    .filter(F.col("operating_day").between(F.date_sub(max_op_date, 6), max_op_date))
    .join(F.broadcast(tl_mbc_stops_df), on="bpuic", how="inner")
    .select("operating_day", "trip_id", "operator_id", "product_id", "transport", "bpuic", "stop_name",
            "arr_time", "arr_actual", "arr_status", "dep_time", "dep_actual", "dep_status")
    .join(F.broadcast(lmr_gares_df), on="operator_id", how="inner")
    .withColumn("agency_id", F.split(F.col("operator_id"), ":")[1])
)

istdaten_filtered = istdaten_filtered.persist(StorageLevel.MEMORY_AND_DISK)
print(istdaten_filtered.count())

# %% [markdown]
# The hint lets us know there may be istdaten trips with missing actual arrival times. It could also be worth ensuring the SBB timetable properly reports arrival times as well.

# %%
(stop_times_recent
    .groupBy(F.col("route_desc"))
    .agg(
        F.sum(F.col("arrival_time").isNull().cast("int")).alias("null_arrival_time"),
        F.sum((~F.col("arrival_time").isNull()).cast("int")).alias("non_null_arrival_time")
    )
    .orderBy(F.col("route_desc"))
).show()

# %%
(istdaten_filtered
    .groupBy(F.col("transport"))
    .agg(
        F.sum(F.col("arr_actual").isNull().cast("int")).alias("null_arrival_time"),
        F.sum((~F.col("arr_actual").isNull()).cast("int")).alias("non_null_arrival_time")
    )
    .orderBy(F.col("transport"))
).show()

# %% [markdown]
# We notice that there is indeed a discrepancy in reporting. Thankfully, istdaten data also has expected, scheduled arrival and departure times. Note that transport and route_desc match nearly perfectly, with the exception of "EXB" missing in the filtered istdaten table, which is a good sign our initial filtering put us on the right track.
#
# Over the next 4 tables, we'd like to explore the combinations of null/non-null scheduled/actual arrival/departure times, to explore if the missing data comes from a complete lack of data, misreporting, or other.

# %%
print("Expected vs actual arrival:")
(istdaten_filtered
 .groupBy(F.col("transport"))
 .agg(
     F.sum(((F.col("arr_time").isNull()) & (F.col("arr_actual").isNull())).cast("int")).alias("null_sched, null_act"),
     F.sum(((F.col("arr_time").isNull()) & (~F.col("arr_actual").isNull())).cast("int")).alias("null_sched, non_null_act"),
     F.sum(((~F.col("arr_time").isNull()) & (F.col("arr_actual").isNull())).cast("int")).alias("non_null_sched, null_act"),
     F.sum(((~F.col("arr_time").isNull()) & (~F.col("arr_actual").isNull())).cast("int")).alias("non_null_sched, non_null_act"),
 )).show()

# %%
print("Expected vs actual departure:")
(istdaten_filtered
 .groupBy(F.col("transport"))
 .agg(
     F.sum(((F.col("dep_time").isNull()) & (F.col("dep_actual").isNull())).cast("int")).alias("null_sched, null_act"),
     F.sum(((F.col("dep_time").isNull()) & (~F.col("dep_actual").isNull())).cast("int")).alias("null_sched, non_null_act"),
     F.sum(((~F.col("dep_time").isNull()) & (F.col("dep_actual").isNull())).cast("int")).alias("non_null_sched, null_act"),
     F.sum(((~F.col("dep_time").isNull()) & (~F.col("dep_actual").isNull())).cast("int")).alias("non_null_sched, non_null_act"),
 )).show()

# %%
print("Expected arrival vs departure:")
(istdaten_filtered
 .groupBy(F.col("transport"))
 .agg(
     F.sum(((F.col("arr_time").isNull()) & (F.col("dep_time").isNull())).cast("int")).alias("null_arr, null_dep"),
     F.sum(((F.col("arr_time").isNull()) & (~F.col("dep_time").isNull())).cast("int")).alias("null_arr, non_null_dep"),
     F.sum(((~F.col("arr_time").isNull()) & (F.col("dep_time").isNull())).cast("int")).alias("non_null_arr, null_dep"),
     F.sum(((~F.col("arr_time").isNull()) & (~F.col("dep_time").isNull())).cast("int")).alias("non_null_arr, non_null_dep"),
 )).show()

# %%
print("Actual arrival vs departure:")
(istdaten_filtered
 .groupBy(F.col("transport"))
 .agg(
     F.sum(((F.col("arr_actual").isNull()) & (F.col("dep_actual").isNull())).cast("int")).alias("null_arr, null_dep"),
     F.sum(((F.col("arr_actual").isNull()) & (~F.col("dep_actual").isNull())).cast("int")).alias("null_arr, non_null_dep"),
     F.sum(((~F.col("arr_actual").isNull()) & (F.col("dep_actual").isNull())).cast("int")).alias("non_null_arr, null_dep"),
     F.sum(((~F.col("arr_actual").isNull()) & (~F.col("dep_actual").isNull())).cast("int")).alias("non_null_arr, non_null_dep"),
 )).show()

# %% [markdown]
# We notice a trend: if there is no scheduled arrival or departure time, then there is no actual time. Moreover, there seems to be the same amount of stop_times lacking information on both arrival and departure time. This can be explained by the fact that those stops must be the starting and ending points of trips: the former would lack information on arrival and the latter on departure. Thus, we need to make sure when we join the dataframes that we include both of those cases, for more accurate matching.
#
# Beyond that, there seem to be a noticeable amount of "R" (Regio) trips missing actual timestamps. However, since istdaten provides us with scheduled arrival and departure times, we can use those to match timetable trips, which will include those missing live data.

# %% [markdown]
# When we try to match istdaten and timetable trips, we need to make sure that the day of the week is the correct one as well. If a weekday-only istdaten trip is the same stops and time-wise as a week-end-only timetable trip, we still cannot validate it.
#
# For the timetable, we can access this information from sbb.calendar, joining on the service_id of the trip. For istdaten data, the scheduled times are in timestamp_ntz format, which also provides us with the date, from which we can extrapolate the weekday.
#
# We also discovered earlier in the assignment that some timetable trips go beyond the 24-hour range. To mitigate this, we can simply apply a modulo operation on the hour; this shouldn't affect results, since we are given by calendar which day the trip is valid for.

# %%
stop_times_recent = (stop_times_recent
    .join(F.broadcast(spark.table("iceberg.sbb.calendar").filter(F.col("pub_date") == max_pub_date)),
        on="service_id", how="inner")
    # Modulo operation on the hour of the timestamp
    .withColumn("arrival_time_24h",
                F.expr("lpad(cast(split(arrival_time, ':')[0] % 24 as string), 2, '0') || ':' || split(arrival_time, ':')[1] || ':' || split(arrival_time, ':')[2]"))
    .withColumn("departure_time_24h",
                F.expr("lpad(cast(split(departure_time, ':')[0] % 24 as string), 2, '0') || ':' || split(departure_time, ':')[1] || ':' || split(departure_time, ':')[2]"))
                    )

stop_times_recent = stop_times_recent.persist(StorageLevel.MEMORY_AND_DISK)
print(stop_times_recent.count())

# %%
istdaten_filtered = (istdaten_filtered
    # arr_time currently formatted as YYYY-MM-DD HH:mm:ss, convert to timestamp for comparison later
    .withColumn("arrival_time", F.date_format(F.col("arr_time"), "HH:mm:ss"))
    .withColumn("departure_time", F.date_format(F.col("dep_time"), "HH:mm:ss"))
    # Give the day of the week
    .withColumn("arr_day", F.lower(F.date_format(F.col("arr_time"), "EEEE")))
    .withColumn("dep_day", F.lower(F.date_format(F.col("dep_time"), "EEEE")))
                    )

istdaten_filtered = istdaten_filtered.persist(StorageLevel.MEMORY_AND_DISK)
print(istdaten_filtered.count())

# %% [markdown]
# Conducting a preliminary match on time, where transport arrives and leave at the same time from the same stop:

# %%
# %%time

"""
Preliminary match on time: match trips where the transport is meant to arrive (ist.arrival_time gives expected/scheduled)
and depart at the same stop, operated by the same operator in both cases. Accounts for starting and ending points.
"""

time_matched_df = (
    istdaten_filtered.alias("ist")
    .join(stop_times_recent.alias("s"),
        # Same stop
        ((F.col("ist.bpuic") == F.col("s.bpuic")) &
        # Same arrival and departures
        (F.col("ist.arrival_time").isNull() | (F.col("ist.arrival_time") == F.col("s.arrival_time_24h"))) & 
        (F.col("ist.departure_time").isNull() | (F.col("ist.departure_time") == F.col("s.departure_time_24h"))) &
        # Same operator
        (F.col("ist.agency_id") == F.col("s.agency_id")))
        )
    .select("ist.operating_day",
            F.col("ist.trip_id").alias("ist_trip_id"),
            F.col("s.trip_id").alias("tt_trip_id"),
            "s.stop_sequence",
            F.col("ist.arrival_time").alias("ist_arrival_time"),
            F.col("s.arrival_time").alias("tt_arrival_time"),
            "ist.stop_name",
            "ist.bpuic",
            "s.stop_id",
            "ist.transport",
            "ist.dep_day",

            # Check whether the Istdaten departure day is valid in the SBB calendar.
            F.coalesce(*[
                F.when(F.col("ist.dep_day") == day, F.col(f"s.{day}"))
                for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            ]).alias("day_matches_calendar")
        )
)

time_matched_df = time_matched_df.persist(StorageLevel.MEMORY_AND_DISK)
print(time_matched_df.count())

# %%
stop_times_recent.unpersist()
istdaten_filtered.unpersist()
# tl_mbc_stops_df.unpersist()

# %% [markdown]
# We want to make sure that the trips from istdaten matched with the timetable trips are running on days that are on days validated by the timetable. If a Monday istdaten trip has the same schedule as a timetable trip that is only allowed on Saturdays, then it can not be allowed. However, some trips can last overnight; a trip might start on Friday night and finish Saturday morning. Therefore, we will be looking at the first stop in a given trip and match the day of operation there, as it's possible that later stops in istdaten will be a different day not approved by the timetable.

# %%
# %%time

# Group the df into trip_ids and only retain the first row for each combination, i.e. the first stop in the sequence
# thanks to sbb.stop_times
first_stop_seq = (
    time_matched_df
    .groupBy("ist_trip_id", "tt_trip_id")
    .agg(F.min("stop_sequence").alias("min_stop_sequence"))
)

valid_pairs = (
    time_matched_df
    .join(first_stop_seq, on=["ist_trip_id", "tt_trip_id"], how="inner")
    # Retain the first stop, avoid unnecessary duplication of pairs
    .filter(F.col("stop_sequence") == F.col("min_stop_sequence"))
    # Only keep combinations where the istdaten trip is on a day allowed by the timetable
    .filter(F.col("day_matches_calendar") == True)
    .select("ist_trip_id", "tt_trip_id")
    .distinct()
)

valid_pairs = valid_pairs.persist(StorageLevel.MEMORY_AND_DISK)
print(valid_pairs.count())


# %% [markdown]
# We can now restrict time_matched_df even further by only keeping valid pairs of days of the week

# %%
# %%time
day_matched_df = time_matched_df.join(valid_pairs, on=["ist_trip_id", "tt_trip_id"], how="inner")

# day_matched_df = time_matched_df.join(first_stop_valid.select("ist_trip_id", "tt_trip_id"), on=["ist_trip_id", "tt_trip_id"], how="inner")
day_matched_df = day_matched_df.persist(StorageLevel.MEMORY_AND_DISK)
print(day_matched_df.count())

# %%
time_matched_df.unpersist()
valid_pairs.unpersist()

# %% [markdown]
# Some transports might arrive/depart at the same stop at the same time (i.e. two buses scheduled to stop by the train
# station simultaneously). Trips are sequences of stops, therefore, we will match trips by matching their sequences.

# %%
# %%time

# Group istdaten trips by stops
ist_trip_stop_count = (
    day_matched_df
    .groupBy("ist_trip_id")
    .agg(F.countDistinct("bpuic").alias("n_stops_matched"))
)

# Group where istdaten and timetable trips share the same stop
matched_trip_stop_count = (
    day_matched_df
    .groupBy("ist_trip_id", "tt_trip_id")
    .agg(F.countDistinct("bpuic").alias("n_trips_matched"))
)

# Only keep sequences where stops are the same along the whole trip. countDistinct is sufficient since we group by trip_ids,
# therefore if the length along a single grouping of stops is the same as the length of a double grouping including that
# same grouping, then they're equal
valid_matches = (
    matched_trip_stop_count
    .join(ist_trip_stop_count, on="ist_trip_id", how="inner")
    .filter(F.col("n_trips_matched") == F.col("n_stops_matched"))
    .select("ist_trip_id", "tt_trip_id")
    .distinct()
)


# %%
# %%time
# From the previously filtered matched df, only keep the trips with valid pairs between istdaten and timetable

matched_df = (
    day_matched_df
    .join(F.broadcast(valid_matches), on=["ist_trip_id", "tt_trip_id"], how="inner")
    .distinct()
    .select("ist_trip_id", "tt_trip_id")
)

matched_df = matched_df.persist(StorageLevel.MEMORY_AND_DISK)
print(matched_df.count())

# %%
day_matched_df.unpersist()

# %% [markdown]
# We have matched_df, the final matching of timetable and istdaten data trips:

# %%
# %%time
matched_df.show(30, truncate=False)

# %%
matched_df.unpersist()

# %% [markdown]
# Clear all the persisted dataframes:

# %%
spark.catalog.clearCache()

# %% [markdown]
# ---
# ## PART IV: SBB Delay Model building (20 points)
#
# In the final segment of this assignment, you will address the problem of predicting SBB delays (in minutes) within the Lausanne region.
#
# You should restrict the analysis to:
#
# - trips operated by the operators identified in Part IIa, and
# - stops served by TL and MBC, as identified in Part IIb.
#
# This is an open-ended problem with multiple valid approaches. We provide a structured sequence of steps to guide you, but beyond that you are expected to work independently. At this stage, you should be comfortable using the Spark API and consulting the documentation to identify the tools you need.
#
# You are encouraged to explore different modeling strategies and to leverage insights and methods developed in earlier parts of the assignment. Creativity, experimentation, and thoughtful justification of your approach are all expected.
#
# 💡 Hints:
#
# - Use at most one year of historical data; older data is unlikely to improve performance significantly.
# - Start with a simple baseline model using a small number of features (e.g., mode of transport, operator and/or time of day), and progressively refine your feature set.

# %% [markdown]
# ### Feature Engineering - 8/20
#
#
# Construct a feature vector for training and testing your model.
#
# Best practices include:
#
# * Data Source Selection and Exploration:
#   - Do not hesitate to reuse the data from Lausanne created earlier. Query the data directly from files into Spark DataFrames.
#   - Explore the data to understand its structure, identifying relevant features and potential issues such as missing or null values.
#
# * Data Sanitization:
#   - Clean up null values and handle any inconsistencies or outliers in the data, as seen for _transport_ and _product_id_.
#
# * Historical Delay Computation:
#   - Utilize the SBB historical istdaten to compute historical delays, incorporating this information into your feature vector.
#   - Experiment with different ways to represent historical delays, such as aggregating delays over different time periods or considering average delays for specific routes or stations.
#
# * Incorporating Additional Data Sources:
#   - Feel free to integrate other relevant data sources into your feature vector (e.g. historical weather data which is available to you).
#   - Explore how these additional features contribute to the predictive power of your model and how they interact with the primary dataset.
#
# * Feature Vector Construction using Spark MLlib:
#   - Utilize [`Spark MLlib`](https://spark.apache.org/docs/latest/ml-features.html). methods to construct the feature vector for your model.
#   - Consider techniques such as feature scaling, transformation, one-hote encoding etc. to enhance the predictive performance of your model.
#

# %% [markdown]
# ### Model building - 5/15
#
# Utilizing the features construct a model capable of predicting delays within the Lausanne region.
#
# To accomplish this task effectively:
#
# * Feature Integration:
#         - Incorporate the features into your modeling pipeline.
#
# * Model Selection and Training:
#         - Explore various machine learning algorithms available in Spark MLlib to identify the most suitable model for predicting delays.
#         - Train the selected model using the feature vectors constructed from the provided data.

# %% [markdown]
# ### Model evaluation - 7/20
#
# * Evaluate the performance of your model
#     * Use appropriate evaluation metrics, provide confidence intervals if possible.
#     * Utilize techniques such as cross-validation to ensure robustness and generalizability of your model.
#     * Do not hesitate to compare different features and model types.
#
# * Interpretation and Iteration:
#     * Interpret the results of your model to gain insights into the factors influencing delays within the Lausanne region.
#     * Iterate on your model by fine-tuning hyperparameters, exploring additional feature engineering techniques, or experimenting with different algorithms to improve predictive performance.
#

# %% [markdown]
# <hr>

# %% [markdown]
# ### 1. Data Exploration and Cleaning

# %% [markdown]
# For Part IV, we rebuild the modeling dataset directly from the original Istdaten table instead of relying on variables created in earlier sections. This makes the delay-modeling section reproducible and easier to rerun independently.
#
# The scope follows the assignment constraints:
#
# - use at most one year of historical Istdaten data;
# - restrict the data to the Lausanne-region operators and stops used earlier;
# - keep only planned trips;
# - remove rows where arrival delay cannot be computed;
# - focus on arrival delay because this is the quantity later needed by the robust journey planner.
#
# For the first modeling version, we use 2025 as a recent complete year. This avoids mixing a full historical year with partial 2026 data.

# %%
# Extra imports for part 4
import os
import time
import subprocess

import pandas as pd
import seaborn as sns

from pyspark.storagelevel import StorageLevel
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
)
from pyspark.ml.evaluation import RegressionEvaluator

# %%
MODEL_START_DATE = "2024-01-01"
MODEL_END_DATE = "2025-12-31"

TL_MBC_OPERATOR_IDS = ["85:151", "85:29", "85:764"]

print(f"Modeling period: {MODEL_START_DATE} to {MODEL_END_DATE}")
print(f"TL/MBC operator ids: {TL_MBC_OPERATOR_IDS}")

# %% [markdown]
# #### Reconstruct the TL/MBC stop scope
#
# In Part II.b, the local modeling area was defined through the stops served by TL/MBC operators. We reconstruct that scope here directly from Istdaten so that Part IV does not depend on previous notebook variables.
#
# This gives us a practical Lausanne-region modeling scope for the delay prediction task.

# %%
tl_mbc_stops_df = (
    spark.table("iceberg.sbb.istdaten")
    .filter(F.col("operating_day").between(MODEL_START_DATE, MODEL_END_DATE))
    .filter(F.col("operator_id").isin(TL_MBC_OPERATOR_IDS))
    .filter(F.col("bpuic").isNotNull())
    .select("bpuic")
    .distinct()
)

n_tl_mbc_stops = tl_mbc_stops_df.count()

print(f"Number of TL/MBC stops in modeling period: {n_tl_mbc_stops}")


# %% [markdown]
# #### Build the raw modeling scope
#
# We first reconstruct the TL/MBC stop set and then filter Istdaten to the records relevant for Part IV. This avoids depending on variables created in previous parts of the notebook.

# %%
def clean_status_col(col):
    return (
        F.when(F.trim(col) == "", "PROGNOSE")
        .otherwise(F.upper(F.trim(col)))
    )
    
tl_mbc_stops_df = (
    spark.table("iceberg.sbb.istdaten")
    .filter(F.col("operating_day").between(MODEL_START_DATE, MODEL_END_DATE))
    .filter(F.col("operator_id").isin(TL_MBC_OPERATOR_IDS))
    .filter(F.col("bpuic").isNotNull())
    .select("bpuic")
    .distinct()
)

print(f"Number of TL/MBC stops: {tl_mbc_stops_df.count()}")

# %%
part4_explore_df = (
    spark.table("iceberg.sbb.istdaten")
    .filter(F.col("operating_day").between(MODEL_START_DATE, MODEL_END_DATE))
    .filter((F.col("failed") == False) | F.col("failed").isNull())
    .filter(F.col("arr_time").isNotNull())
    .filter(F.col("arr_actual").isNotNull())
    .filter(F.col("arr_status").isNotNull())
    .withColumn("clean_arr_status", clean_status_col(F.col("arr_status")))
    .filter(F.col("clean_arr_status") != "UNBEKANNT")
    .join(tl_mbc_stops_df, on="bpuic", how="inner")
    .filter(F.col("operator_id").isin(TL_MBC_OPERATOR_IDS))
    .withColumn("arr_time_ts", F.col("arr_time").cast("timestamp"))
    .withColumn("arr_actual_ts", F.col("arr_actual").cast("timestamp"))
    .withColumn(
        "delay_sec",
        F.col("arr_actual_ts").cast("long") - F.col("arr_time_ts").cast("long")
    )
    .withColumn("delay_min", F.col("delay_sec") / F.lit(60.0))
    .withColumn("year", F.year("operating_day"))
    .withColumn("month", F.month("operating_day"))
    .withColumn("day_of_week", F.dayofweek("operating_day"))
    .withColumn("hour", F.hour("arr_time_ts"))
    .select(
        "operating_day",
        "arr_time_ts",
        "arr_actual_ts",
        "year",
        "month",
        "day_of_week",
        "hour",
        "bpuic",
        "stop_name",
        "trip_id",
        "operator_id",
        "operator_abrv",
        "product_id",
        "transport",
        "line_id",
        "line_text",
        "clean_arr_status",
        "delay_sec",
        "delay_min"
    )
)

part4_explore_df.cache()

print(f"Rows in Part IV exploration dataset: {part4_explore_df.count()}")
part4_explore_df.show(5, truncate=False)

# %% [markdown]
# We check the basic size and coverage of the dataset before modeling. This also tells us whether some categorical features, such as `trip_id`, may be too large for simple one-hot encoding.

# %%
coverage_df = (
    part4_explore_df
    .agg(
        F.count("*").alias("n_rows"),
        F.countDistinct("operating_day").alias("n_days"),
        F.min("operating_day").alias("min_day"),
        F.max("operating_day").alias("max_day"),
        F.countDistinct("bpuic").alias("n_stops"),
        F.countDistinct("trip_id").alias("n_trips"),
        F.countDistinct("line_text").alias("n_lines"),
        F.countDistinct("operator_id").alias("n_operators"),
        F.countDistinct("product_id").alias("n_product_ids"),
        F.countDistinct("transport").alias("n_transport_values")
    )
)

coverage_df.show(truncate=False)

# %% [markdown]
# #### Transport/product consistency
#
# Before using transport and product_id as features, we inspect their combinations. This reuses the issue found earlier: the same transport mode may appear with inconsistent labels or product ids.

# %%
transport_product_scope_df = (
    part4_explore_df
    .withColumn("transport_raw", F.trim(F.upper(F.col("transport"))))
    .withColumn("product_id_raw", F.trim(F.upper(F.col("product_id"))))
    .groupBy("transport_raw", "product_id_raw")
    .agg(F.count("*").alias("frequency"))
    .orderBy(F.desc("frequency"))
)

transport_product_scope_df.show(30, truncate=False)

# %% [markdown]
# **Result.** In the selected TL/MBC scope, the transport/product mapping is already quite clean: buses map to BUS, metro to METRO, regional trains to ZUG, and night buses also to BUS. We still keep a standardization step later to make the pipeline robust and consistent with the issues identified earlier in Part I.e.

# %% [markdown]
# #### Delay distribution

# %%
delay_summary_df = (
    part4_explore_df
    .agg(
        F.count("*").alias("n_rows"),
        F.round(F.mean("delay_min"), 2).alias("mean_delay_min"),
        F.round(F.stddev("delay_min"), 2).alias("std_delay_min"),
        F.round(F.min("delay_min"), 2).alias("min_delay_min"),
        F.round(F.expr("percentile_approx(delay_min, 0.50)"), 2).alias("median_delay_min"),
        F.round(F.expr("percentile_approx(delay_min, 0.90)"), 2).alias("p90_delay_min"),
        F.round(F.expr("percentile_approx(delay_min, 0.95)"), 2).alias("p95_delay_min"),
        F.round(F.expr("percentile_approx(delay_min, 0.99)"), 2).alias("p99_delay_min"),
        F.round(F.max("delay_min"), 2).alias("max_delay_min")
    )
)

delay_summary_df.show(truncate=False)

# %%
delay_quality_df = (
    part4_explore_df
    .agg(
        F.count("*").alias("n_rows"),
        F.sum(F.when(F.col("delay_min") < 0, 1).otherwise(0)).alias("n_early_arrivals"),
        F.sum(F.when(F.col("delay_min") > 0, 1).otherwise(0)).alias("n_late_arrivals"),
        F.sum(F.when(F.col("delay_min") > 30, 1).otherwise(0)).alias("n_delay_over_30min"),
        F.sum(F.when(F.col("delay_min") > 60, 1).otherwise(0)).alias("n_delay_over_60min")
    )
    .withColumn("early_arrival_pct", F.round(F.col("n_early_arrivals") / F.col("n_rows") * 100, 2))
    .withColumn("delay_over_30min_pct", F.round(F.col("n_delay_over_30min") / F.col("n_rows") * 100, 2))
    .withColumn("delay_over_60min_pct", F.round(F.col("n_delay_over_60min") / F.col("n_rows") * 100, 2))
)

delay_quality_df.show(truncate=False)

# %% [markdown]
# #### Delay by hour/day/month
#
# We aggregate delays by scheduled arrival hour. This helps decide whether time-of-day features are useful for the baseline model.

# %%
delay_by_hour_df = (
    part4_explore_df
    .groupBy("hour")
    .agg(
        F.count("*").alias("frequency"),
        F.round(F.avg("delay_min"), 2).alias("mean_delay_min"),
        F.round(F.expr("percentile_approx(delay_min, 0.5)"), 2).alias("median_delay_min"),
        F.round(F.expr("percentile_approx(delay_min, 0.9)"), 2).alias("p90_delay_min")
    )
    .orderBy("hour")
)

delay_by_hour_df.show(24)

# %%
hourly_pdf = delay_by_hour_df.toPandas()

plt.figure(figsize=(10, 5))
plt.plot(hourly_pdf["hour"], hourly_pdf["median_delay_min"], marker="o", label="Median delay")
plt.plot(hourly_pdf["hour"], hourly_pdf["p90_delay_min"], marker="o", label="90th percentile delay")
plt.xlabel("Scheduled arrival hour")
plt.ylabel("Arrival delay (minutes)")
plt.title("Arrival delay by hour of day")
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# %% [markdown]
# The 90th percentile delay is highest during the early morning and especially around the evening peak, while the median delay remains much lower. This supports using time-of-day features in the first model.

# %%
fig, axs = plt.subplots(1,3, figsize=(14,3), sharey=True)
plt.ylabel('Delay in minutes')

plot_sample = part4_explore_df[part4_explore_df.year==2025].select(['month', 'day_of_week', 'hour', 'delay_min']).sample(0.01, seed=490).toPandas()
baseline = plot_sample.delay_min.mean()

axs[0].hlines(baseline, xmin=1, xmax=12)
sns.lineplot(data=plot_sample, x='month', y='delay_min', ax=axs[0], color='C1', marker='o', seed=490)
axs[0].set_xlabel('Month')
axs[0].legend(['Mean delay', 'Mean delay by month'])

axs[1].hlines(baseline, xmin=1, xmax=7)
sns.lineplot(data=plot_sample, x='day_of_week', y='delay_min', ax=axs[1], color='C1', marker='o', seed=490)
axs[1].set_xlabel('Day of the week')
axs[1].legend(['Mean delay', 'Mean delay by day'])

axs[2].hlines(baseline, xmin=0, xmax=23)
sns.lineplot(data=plot_sample, x='hour', y='delay_min', ax=axs[2], color='C1', marker='o', seed=490)
axs[2].set_xlabel('Hour of the day')
axs[2].legend(['Mean delay', 'Mean delay by hour'])

plt.title("Mean delay by month, day and hour")
plt.show()


# %% [markdown]
# It is clear that delays vary by month, day and time of day in a non-linear fashion. It therefore makes sense to one-hot encode those features to enable the models to learn good representations of their importance. We also note that there is no mean delay for the months of August and September, an omission that we'll have to keep in mind when training and evalutating our models.

# %% [markdown]
#  #### Cleaning

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# The exploration dataset already removed rows where arrival delay could not be computed and ignored unknown arrival-status records. We now only apply the cleaning needed before feature engineering.
#
# We reuse the same transport/product standardization logic from Part I.e: trim strings, uppercase labels, map inconsistent abbreviations, and enforce a consistent product category.
#
# For the modeling target, we keep the raw delay but also create label_delay_min. Negative delays are set to 0 because early arrivals do not help the robust planner, and very large delays are capped at 60 minutes to reduce the effect of extreme outliers.

# %%
def clean_part4_modeling_data(df):
    """
    Standardize transport fields and create the capped non-negative delay label
    used for Part IV modeling.
    """
    cleaned_df = (
        df
        .withColumn("clean_transport", F.trim(F.upper(F.col("transport"))))
        .withColumn("clean_product_id", F.trim(F.upper(F.col("product_id"))))
        .withColumn(
            "clean_transport",
            F.when(F.col("clean_transport") == "BUS", "B")
            .otherwise(F.col("clean_transport"))
        )
        .withColumn(
            "clean_product_id",
            F.when(F.col("clean_transport").isin("CAR", "EV", "KB", "B", "EXB", "BN", "BP", "RUB"), "BUS")
            .when(F.col("clean_transport") == "T", "TRAM")
            .when(F.col("clean_transport") == "M", "METRO")
            .when(F.col("clean_transport").isin("BAT", "FAE"), "SCHIFF")
            .when(F.col("clean_transport").isin("GB", "PB", "SL", "ASC", "FUN", "CC"), "ZAHNRADBAHN")
            .when(
                F.col("clean_transport").isin(
                    "ZUG", "TGV", "EST", "IC", "ICE", "RJX", "EC", "IR", "IRE", "ATZ",
                    "ARZ", "NJ", "EN", "RB", "TER", "RE", "R", "PE", "S", "SN", "EXT"
                ),
                "ZUG"
            )
            .when((F.col("clean_transport") == "") & (F.col("clean_product_id") == "ZUG"), "ZUG")
            .otherwise(F.col("clean_product_id"))
        )
        .filter(F.col("trip_id").isNotNull() & (F.trim(F.col("trip_id")) != ""))
        .filter(F.col("operator_id").isNotNull() & (F.trim(F.col("operator_id")) != ""))
        .filter(F.col("clean_transport").isNotNull() & (F.col("clean_transport") != ""))
        .filter(F.col("clean_product_id").isNotNull() & (F.col("clean_product_id") != ""))
        .withColumn("raw_delay_min", F.col("delay_min"))
        .withColumn(
            "label_delay_min",
            F.least(
                F.greatest(F.col("delay_min"), F.lit(0.0)),
                F.lit(60.0)
            )
        )
    )

    return cleaned_df


# %%
part4_clean_df = clean_part4_modeling_data(part4_explore_df)

part4_clean_df.cache()

print(f"Rows before cleaning: {part4_explore_df.count()}")
print(f"Rows after cleaning:  {part4_clean_df.count()}")

part4_clean_df.select(
    "bpuic",
    "trip_id",
    "operator_id",
    "clean_product_id",
    "clean_transport",
    "hour",
    "day_of_week",
    "raw_delay_min",
    "label_delay_min"
).show(10, truncate=False)

# %%
cleaning_check_df = (
    part4_clean_df
    .agg(
        F.count("*").alias("n_rows"),
        F.round(F.avg("raw_delay_min"), 2).alias("mean_raw_delay_min"),
        F.round(F.avg("label_delay_min"), 2).alias("mean_label_delay_min"),
        F.round(F.expr("percentile_approx(label_delay_min, 0.5)"), 2).alias("median_label_delay_min"),
        F.round(F.expr("percentile_approx(label_delay_min, 0.9)"), 2).alias("p90_label_delay_min"),
        F.round(F.max("label_delay_min"), 2).alias("max_label_delay_min")
    )
)

cleaned_transport_product_df = (
    part4_clean_df
    .groupBy("clean_transport", "clean_product_id")
    .agg(F.count("*").alias("frequency"))
    .orderBy(F.desc("frequency"))
)

cleaning_check_df.show(truncate=False)
cleaned_transport_product_df.show(30, truncate=False)

# %% [markdown]
#  #### Historical delay patterns
#
#  We inspect historical delay aggregates by stop, line, and time.
#  These tables are descriptive only. For modeling, the same aggregates must be
#  recomputed inside each temporal fold using only the training period.

# %%
hist_delay_explore_df = (
    part4_clean_df
    .groupBy("bpuic", "line_text", "hour")
    .agg(
        F.count("*").alias("n_obs"),
        F.round(F.avg("label_delay_min"), 3).alias("mean_delay_min"),
        F.round(F.expr("percentile_approx(label_delay_min, 0.50)"), 3).alias("p50_delay_min"),
        F.round(F.expr("percentile_approx(label_delay_min, 0.90)"), 3).alias("p90_delay_min"),
        F.round(F.expr("percentile_approx(label_delay_min, 0.95)"), 3).alias("p95_delay_min"),
    )
    .orderBy(F.desc("p90_delay_min"))
)

hist_delay_explore_df.show(30, truncate=False)

# %% [markdown]
# ### 2. External Data Sources

# %% [markdown]
# #### Weather Data
#
# We enrich the modeling dataset with hourly weather observations. Each stop is matched to its nearest weather station, and weather values are joined by the scheduled arrival hour.

# %%
# Weather history source
SILVER_BASE_PATH = f"{hadoopFS}/data/com-490/silver"
WEATHER_HISTORY_PATH = f"{SILVER_BASE_PATH}/weather/history"

weather_history_df = (
    spark.read
    .format("iceberg")
    .load(WEATHER_HISTORY_PATH)
)

weather_history_df.show(5, truncate=False)
weather_history_df.printSchema()

# %%
# Weather stations source
WEATHER_STATIONS_PATH = f"{SILVER_BASE_PATH}/weather/stations"

stations_df = spark.read.parquet(WEATHER_STATIONS_PATH)

stations_clean_df = (
    stations_df
    .filter(F.col("ws_name") != "Name,City,Canton,ID,Active,lat,lon")
    .withColumn("parts", F.split(F.col("ws_name"), ","))
    .select(
        F.col("parts")[0].alias("station_name"),
        F.col("parts")[1].alias("station_city"),
        F.col("parts")[2].alias("station_canton"),
        F.col("parts")[3].alias("station_id"),
        F.col("parts")[4].cast("boolean").alias("station_active"),
        F.col("parts")[5].cast("double").alias("station_lat"),
        F.col("parts")[6].cast("double").alias("station_lon"),
    )
    .filter(F.col("station_id").isNotNull())
    .filter(F.col("station_lat").isNotNull())
    .filter(F.col("station_lon").isNotNull())
)

stations_clean_df.show(5, truncate=False)
stations_clean_df.printSchema()

# %%
# null / non-null exploration for weather history
weather_nulls_df = (
    weather_history_df
    .agg(*[
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in weather_history_df.columns
    ])
)

weather_counts_df = (
    weather_history_df
    .agg(F.count("*").alias("n_rows"))
)

n_rows = weather_counts_df.collect()[0]["n_rows"]

weather_null_summary_df = (
    weather_nulls_df
    .select([
        F.struct(
            F.lit(c).alias("column"),
            F.col(c).alias("null_count"),
            F.round((F.col(c) / F.lit(n_rows)) * 100, 2).alias("null_pct")
        ).alias(c)
        for c in weather_history_df.columns
    ])
    .select(F.explode(F.array(*weather_history_df.columns)).alias("x"))
    .select("x.column", "x.null_count", "x.null_pct")
    .withColumn("non_null_pct", F.round(F.lit(100.0) - F.col("null_pct"), 2))
    .orderBy(F.col("null_pct").desc())
)

print(f"Weather history rows: {n_rows:,}")
weather_null_summary_df.show(60, truncate=False)

# %% [markdown]
# Based on the missing-value analysis, we removed weather columns that were almost entirely empty or irrelevant for urban public transport, such as snow, water, wave, swell, and textual qualifier fields. We kept compact numerical weather variables with high coverage and plausible impact on delays, such as temperature, humidity, pressure, wind, dew point, and UV index. Hourly precipitation is kept separately despite its missing values, because rain may affect bus and tram delays and can later be handled with a missingness flag.

# %%
# Filter weather history to the selected useful columns.
# Cleaning/imputation will be handled later
# Joining with the training dataset will be handled later in feature engineering.

selected_weather_cols = [
    "location_id",
    "valid_time_gmt",
    "temp",
    "feels_like",
    "rh",
    "pressure",
    "wspd",
    "wdir",
    "dewpt",
    "heat_index",
    "wc",
    "uv_index",
    "precip_hrly"
]

weather_filtered_df = (
    weather_history_df
    .select(*selected_weather_cols)
)

weather_filtered_df.show(5, truncate=False)
weather_filtered_df.printSchema()


# %% [markdown]
# #### External calendar data
#
# We extend the SBB delay dataset with a small external calendar table for Canton Vaud. It contains public holidays and Vaud school-holiday periods for 2025 and 2026, collected from official Canton Vaud calendar pages.
#
# The CSV is stored as a period table with the columns `start_date`, `end_date`, `kind`, `name`, and `source`. During feature engineering, these periods are expanded into daily flags and joined to Istdaten using `operating_day`. This keeps the pipeline reproducible and avoids depending on external APIs at runtime.
#
# The file is located at:
#
# `data/vaud_calendar_periods.csv`
#
# Sources:
# - Canton Vaud, public holidays calendar 2024
# - Canton Vaud, school holidays calendar 2025
# - Canton Vaud, school holidays calendar 2026

# %%
def build_calendar_features(spark, calendar_csv_path, start_date, end_date):
    """
    Builds daily public-holiday and school-holiday features from a period-based
    Vaud calendar CSV for the selected modeling date range.
    """ 
    periods = (
        spark.read.option("header", True).csv(calendar_csv_path)
        .withColumn("start_date", F.to_date("start_date"))
        .withColumn("end_date", F.to_date("end_date"))
        .filter(F.col("end_date") >= F.lit(start_date))
        .filter(F.col("start_date") <= F.lit(end_date))
    )

    days = (
        periods
        .select(
            F.explode(F.sequence("start_date", "end_date")).alias("operating_day"),
            "kind"
        )
        .filter(F.col("operating_day").between(start_date, end_date))
        .dropDuplicates()
    )

    base = (
        days.groupBy("operating_day")
        .pivot("kind", ["public_holiday", "school_holiday"])
        .count()
        .fillna(0)
        .withColumnRenamed("public_holiday", "is_public_holiday")
        .withColumnRenamed("school_holiday", "is_school_holiday")
    )

    def shifted_flag(df, kind, date_col, shift_days, flag_name):
        return (
            df.filter(F.col("kind") == kind)
            .select(F.date_add(F.col(date_col), shift_days).alias("operating_day"))
            .filter(F.col("operating_day").between(start_date, end_date))
            .distinct()
            .withColumn(flag_name, F.lit(1))
        )

    before_after = [
        shifted_flag(days, "public_holiday", "operating_day", -1, "is_day_before_public_holiday"),
        shifted_flag(days, "public_holiday", "operating_day",  1, "is_day_after_public_holiday"),
        shifted_flag(periods, "school_holiday", "start_date", -1, "is_day_before_school_holiday"),
        shifted_flag(periods, "school_holiday", "end_date",    1, "is_day_after_school_holiday"),
    ]

    bridge = (
        shifted_flag(days, "public_holiday", "operating_day", 1, "is_bridge_day")
        .filter(F.dayofweek("operating_day") == 6)  # Friday after Thursday holiday
        .unionByName(
            shifted_flag(days, "public_holiday", "operating_day", -1, "is_bridge_day")
            .filter(F.dayofweek("operating_day") == 2)  # Monday before Tuesday holiday
        )
        .distinct()
    )

    features = base
    for extra in before_after + [bridge]:
        features = features.join(extra, on="operating_day", how="full")

    calendar_numeric_cols = [
        "is_public_holiday",
        "is_school_holiday",
        "is_day_before_public_holiday",
        "is_day_after_public_holiday",
        "is_day_before_school_holiday",
        "is_day_after_school_holiday",
        "is_bridge_day",
    ]

    return features.fillna(0, subset=calendar_numeric_cols), calendar_numeric_cols


# %%
# Copy the local calendar CSV to HDFS so Spark can read it on the cluster.
LOCAL_CALENDAR_CSV = "data/vaud_calendar_periods.csv"

if not os.path.exists(LOCAL_CALENDAR_CSV):
    raise FileNotFoundError(f"Could not find calendar CSV at {LOCAL_CALENDAR_CSV}")

HDFS_CALENDAR_DIR = f"/user/{username}/data"
HDFS_CALENDAR_CSV = f"{HDFS_CALENDAR_DIR}/vaud_calendar_periods.csv"

subprocess.run(["hdfs", "dfs", "-mkdir", "-p", HDFS_CALENDAR_DIR], check=True)
subprocess.run(["hdfs", "dfs", "-put", "-f", LOCAL_CALENDAR_CSV, HDFS_CALENDAR_CSV], check=True)

CALENDAR_CSV_PATH = f"{hadoopFS}{HDFS_CALENDAR_CSV}"

print(f"Uploaded calendar CSV to HDFS: {CALENDAR_CSV_PATH}")

calendar_features_df, calendar_numeric_cols = build_calendar_features(
    spark=spark,
    calendar_csv_path=CALENDAR_CSV_PATH,
    start_date=MODEL_START_DATE,
    end_date=MODEL_END_DATE,
)

# %% [markdown]
# **Result.** The cleaned dataset keeps the same modeling scope but standardizes the categorical transport fields and creates a safer target for prediction. The model will train on `label_delay_min`, while `raw_delay_min` is kept for comparison and interpretation.

# %%
# Clean selected weather history.
weather_clean_df = (
    weather_filtered_df
    .filter(F.col("location_id").isNotNull() & F.col("valid_time_gmt").isNotNull())
    .withColumn("weather_hour", F.date_trunc("hour", F.from_utc_timestamp("valid_time_gmt", "Europe/Zurich")))
    .withColumn("precip_hrly_missing", F.when(F.col("precip_hrly").isNull(), 1.0).otherwise(0.0))
    .withColumn("precip_hrly", F.coalesce(F.col("precip_hrly"), F.lit(0.0)))
    .withColumn("is_rainy", F.when(F.col("precip_hrly") > 0, 1.0).otherwise(0.0))
    .select(
        F.col("location_id").alias("station_id"),
        "weather_hour",
        "temp", "feels_like", "rh", "pressure", "wspd", "wdir",
        "dewpt", "heat_index", "wc", "uv_index",
        "precip_hrly", "precip_hrly_missing", "is_rainy"
    )
    .dropDuplicates(["station_id", "weather_hour"])
)

weather_clean_df.show(5, truncate=False)

# %% [markdown]
# ### 4. Feature Engineering
#
# In this section, we construct the feature vector for delay prediction using Spark MLlib.
#
# The initial prototype used a rich set of categorical identifiers, including `trip_id` and `clean_arr_status`, together with stop, operator, product, transport, line, and time-based features. However, after one-hot encoding, this representation produced a very large sparse feature vector with 54,201 features. Most of this increase came from high-cardinality identifiers, especially trip_id.
#
# Although this version was technically trainable, it is not ideal for our use case. The goal of the delay model is not to memorize the behavior of individual historical trips, but to learn general delay patterns that can be reused inside the robust journey planner. In particular, trip_id is too specific and may encourage overfitting to past schedules. Similarly, clean_arr_status describes how the actual arrival time was measured, which is not a reliable planning-time feature for future journeys.
#
# We therefore tested a reduced feature representation that removes trip_id and clean_arr_status, while keeping features that are more stable and meaningful at planning time. After the later additions, the final feature set also includes bus-specific weather features, external calendar features, and fold-safe historical delay aggregates.
#
# The feature vector contains:
#
# - stop id: `bpuic`
# - line name: `line_text`
# - scheduled arrival hour: `hour`
# - day of week: `day_of_week`
# - month: `month`
# - bus-specific weather features
# - calendar features for public holidays, school holidays, bridge days, and days around holidays
# - historical delay aggregate features computed inside each temporal fold
#
# Categorical columns are indexed and one-hot encoded. Numeric columns are added directly to the feature vector. The target variable is `label_delay_min`.

# %%
feature_input_df = (
    part4_clean_df
    .select(
        "operating_day",
        "bpuic",
        "trip_id",
        "operator_id",
        "clean_product_id",
        "clean_transport",
        "line_text",
        "clean_arr_status",
        "hour",
        "day_of_week",
        "month",
        "label_delay_min"
    )
    .filter(F.col("label_delay_min").isNotNull())
    .filter(F.col("hour").isNotNull())
    .filter(F.col("day_of_week").isNotNull())
    .filter(F.col("month").isNotNull())
)

feature_input_df.cache()

print(f"Rows used for feature engineering: {feature_input_df.count()}")
feature_input_df.show(5, truncate=False)

# %% [markdown]
# #### Joining weather features

# %%
# Match each stop to the nearest weather station and join hourly weather by scheduled arrival time.
max_pub_date = (
    spark.table("iceberg.sbb.stops")
    .select(F.max("pub_date").alias("max_pub_date"))
    .collect()[0]["max_pub_date"]
)

model_stops_df = (
    part4_clean_df
    .select("bpuic")
    .distinct()
    .join(
        spark.table("iceberg.sbb.stops")
        .filter(F.col("pub_date") == max_pub_date)
        .withColumn("bpuic", F.substring("stop_id", 1, 7).cast("int"))
        .groupBy("bpuic")
        .agg(
            F.first("stop_name", ignorenulls=True).alias("stop_name"),
            F.avg("stop_lat").alias("stop_lat"),
            F.avg("stop_lon").alias("stop_lon")
        ),
        on="bpuic",
        how="left"
    )
)

stop_station_df = (
    model_stops_df
    .filter(F.col("stop_lat").isNotNull() & F.col("stop_lon").isNotNull())
    .crossJoin(F.broadcast(stations_clean_df))
    .withColumn(
        "distance_km",
        F.expr("""
            6371 * 2 * asin(LEAST(sqrt(
                pow(sin(radians(station_lat - stop_lat) / 2), 2) +
                cos(radians(stop_lat)) * cos(radians(station_lat)) *
                pow(sin(radians(station_lon - stop_lon) / 2), 2)
            ), 1))
        """)
    )
    .withColumn(
        "rn",
        F.row_number().over(
            Window.partitionBy("bpuic").orderBy(F.col("distance_km"))
        )
    )
    .filter(F.col("rn") == 1)
    .select(
        "bpuic",
        F.col("station_id").alias("ws_id"),
        F.col("station_city").alias("city"),
        "distance_km"
    )
)

stop_station_df.show(20, truncate=False)

# %%
# Add weather features to the Part IV modeling dataset.
# Flow: bpuic -> nearest station_id -> weather at scheduled arrival hour.

weather_numeric_cols = [
    "temp", "feels_like", "rh", "pressure", "wspd", "wdir",
    "dewpt", "heat_index", "wc", "uv_index",
    "precip_hrly", "precip_hrly_missing", "is_rainy"
]

weather_bus_cols = [col + "_bus" for col in weather_numeric_cols] # It makes sense to only consider the weather if the transport is a bus

weather_for_join_df = (
    weather_clean_df
    .withColumnRenamed("station_id", "weather_station_id")
)

part4_with_weather_df = (
    part4_clean_df.alias("p")
    .join(
        stop_station_df.select(
            F.col("bpuic").alias("ss_bpuic"),
            F.col("ws_id").alias("nearest_station_id")
        ).alias("s"),
        F.col("p.bpuic") == F.col("s.ss_bpuic"),
        "left"
    )
    .withColumn("arr_hour", F.date_trunc("hour", F.col("p.arr_time_ts")))
    .join(
        weather_for_join_df.alias("w"),
        (F.col("nearest_station_id") == F.col("w.weather_station_id")) &
        (F.col("arr_hour") == F.col("w.weather_hour")),
        "left"
    )
    .drop("ss_bpuic", "weather_station_id", "weather_hour")
    .fillna({col: weather_for_join_df.select(F.mean(weather_for_join_df[col])).first()[0] for col in weather_numeric_cols}) # Fills in missing values with the mean of the column,
    # ensuring that the model can still learn feature importance
)

part4_with_weather_df.select(
    "bpuic", "arr_time_ts", "arr_hour", "nearest_station_id",
    "temp", "rh", "pressure", "wspd", "precip_hrly", "is_rainy", "label_delay_min"
).show(10, truncate=False)

# %% [markdown]
# #### Joining calendar features

# %%
# Add calendar features after weather features dataframe
part4_with_calendar_df = (
    part4_with_weather_df
    .withColumn("operating_day", F.to_date("operating_day"))
    .join(F.broadcast(calendar_features_df), on="operating_day", how="left")
    .fillna(0.0, subset=calendar_numeric_cols)
)

part4_with_calendar_df.cache()

print(f"Rows with weather + calendar features: {part4_with_calendar_df.count():,}")

part4_with_calendar_df.select(
    "operating_day",
    "is_public_holiday",
    "is_school_holiday",
    "is_day_before_public_holiday",
    "is_day_after_public_holiday",
    "is_day_before_school_holiday",
    "is_day_after_school_holiday",
    "is_bridge_day",
    "label_delay_min"
).show(20, truncate=False)

# %%
# Build final modeling input with weather + calendar features

feature_input_df = (
    part4_with_calendar_df
    .withColumns({
        f"{col}_bus": F.when(
            F.col("clean_product_id") == "BUS",
            F.col(col).cast("double")
        ).otherwise(F.lit(0.0))
        for col in weather_numeric_cols
    })
    .filter(F.col("label_delay_min").isNotNull())
    .filter(F.col("hour").isNotNull())
    .filter(F.col("day_of_week").isNotNull())
    .filter(F.col("month").isNotNull())
    .select(
        "bpuic",
        "line_text",
        "hour",
        "day_of_week",
        "month",
        "operating_day",
        *weather_bus_cols,
        *calendar_numeric_cols,
        "label_delay_min"
    )
)

feature_input_df.cache()

print(f"Rows used for feature engineering: {feature_input_df.count():,}")
print("Calendar columns added:")
print(calendar_numeric_cols)

feature_input_df.show(5, truncate=False)

# %% [markdown]
# #### Historical delay aggregate features
#
# Historical delay features are computed inside each temporal fold using only
# the training period. This avoids lookahead leakage.

# %%
HIST_AGG_SPECS = [
    (["bpuic"], "hist_stop"),
    (["line_text"], "hist_line"),
    (["bpuic", "hour"], "hist_stop_hour"),
    (["line_text", "hour"], "hist_line_hour"),
    (["day_of_week", "hour"], "hist_dow_hour"),
]

def add_historical_delay_features(train_df, target_df, label_col="label_delay_min"):
    """
    Build historical delay aggregate features from train_df only,
    then join them into target_df.
    """
    global_stats = (
        train_df
        .agg(
            F.avg(label_col).alias("global_mean"),
            F.expr(f"percentile_approx({label_col}, 0.90)").alias("global_p90"),
            F.expr(f"percentile_approx({label_col}, 0.95)").alias("global_p95"),
        )
        .collect()[0]
    )

    out_df = target_df
    hist_cols = []

    for keys, prefix in HIST_AGG_SPECS:
        agg_df = (
            train_df
            .groupBy(*keys)
            .agg(
                F.count("*").cast("double").alias(f"{prefix}_n"),
                F.avg(label_col).alias(f"{prefix}_mean"),
                F.expr(f"percentile_approx({label_col}, 0.90)").alias(f"{prefix}_p90"),
                F.expr(f"percentile_approx({label_col}, 0.95)").alias(f"{prefix}_p95"),
            )
        )

        out_df = out_df.join(agg_df, on=keys, how="left")

        hist_cols += [
            f"{prefix}_n",
            f"{prefix}_mean",
            f"{prefix}_p90",
            f"{prefix}_p95",
        ]

    fill_values = {}
    for c in hist_cols:
        if c.endswith("_n"):
            fill_values[c] = 0.0
        elif c.endswith("_mean"):
            fill_values[c] = float(global_stats["global_mean"] or 0.0)
        elif c.endswith("_p90"):
            fill_values[c] = float(global_stats["global_p90"] or 0.0)
        elif c.endswith("_p95"):
            fill_values[c] = float(global_stats["global_p95"] or 0.0)

    out_df = out_df.fillna(fill_values)

    return out_df, hist_cols


# %%
# Feature column definitions for the temporal CV pipeline.
# The actual StringIndexer, OneHotEncoder, and VectorAssembler will be fitted inside each fold.

categorical_cols = [
    "bpuic",
    "line_text",
    "hour",
    "day_of_week",
    "month",
]

numeric_cols = weather_bus_cols  + calendar_numeric_cols

label_col = "label_delay_min"

feature_input_df = (
    feature_input_df
    .select(
        "operating_day",
        *categorical_cols,
        *numeric_cols,
        label_col
    )
    .filter(F.col(label_col).isNotNull())
)

feature_input_df.cache()

print(f"Rows available for temporal CV: {feature_input_df.count():,}")
feature_input_df.show(5, truncate=False)


# %%
def build_pipeline(model, categorical_cols, numeric_cols, regression_type="linear"):
    """
    Build a Spark ML pipeline with indexing, one-hot encoding, feature assembly,
    and the selected regression model.
    """
    index_cols = [c + "_index" for c in categorical_cols]
    encoded_cols = [c + "_vec" for c in categorical_cols]

    indexer = StringIndexer(
        inputCols=categorical_cols,
        outputCols=index_cols,
        handleInvalid="keep"
    )

    encoder = OneHotEncoder(
        inputCols=index_cols,
        outputCols=encoded_cols,
        handleInvalid="keep",
        dropLast=(regression_type == "linear")
    )

    assembler = VectorAssembler(
        inputCols=encoded_cols + numeric_cols,
        outputCol="features",
        handleInvalid="keep"
    )

    return Pipeline(stages=[indexer, encoder, assembler, model])


# %%
def prepare_ml_columns(df, categorical_cols, numeric_cols, label_col):
    """
    Cast categorical, numeric, and label columns to Spark ML-compatible types
    and fill missing feature values.
    """
    out_df = df

    for c in categorical_cols:
        out_df = out_df.withColumn(
            c,
            F.coalesce(F.col(c).cast("string"), F.lit("UNKNOWN"))
        )

    for c in numeric_cols:
        out_df = out_df.withColumn(
            c,
            F.coalesce(F.col(c).cast("double"), F.lit(0.0))
        )

    out_df = out_df.withColumn(label_col, F.col(label_col).cast("double"))

    return out_df


# %% [markdown]
# ### 5. Model Selection and Building

# %% [markdown]
# #### Temporal cross-validation strategy
#
# Our initial idea was to use standard random cross-validation, since it is directly supported by Spark MLlib and would give a simple average validation score. However, for delay prediction this would be scientifically misleading: random splits could mean that the model has access to delays of a certain trip and can therefore predict delays simply by checking whether that trip faced delays in the very near past/future. Simple temporal splits, on the other hand, would mean that training and validation/test data sets consist of completely different months, resulting in model underperformance due to the different distribution of the data. This can be especially severe due to One-Hot encoding, since the model would be unable to learn feature importance of unseen months. Since all three periodicities that we explored earlier (hour/day/month) appear to be correlated with the mean delay, we decided to include them as features, train on one-year historical windows, and validate on later seasonal periods. This strategy allows us to test real-world performance of our model, since the actual use case would be predicting delays at unseen future time points.
#
# To avoid lookahead bias, we use a seasonal temporal validation strategy:
#
# - **Winter fold:** train on March 2024–February 2025, validate on December 2025–February 2026.
# - **Spring fold:** train on March 2024–February 2025, validate on March–May 2025.
# - **Summer fold:** train on June 2024–May 2025, validate on June–August 2025.
# - **Autumn fold:** train on September 2024–August 2025, validate on September–November 2025.
#
# In each fold, historical aggregate features are recomputed using only the training period and then joined into the validation period. This keeps the validation data temporally separated from the aggregate-feature computation and avoids lookahead leakage.

# %%
"""
Before running temporal cross-validation, we materialize the feature table and set a checkpoint directory. 
This avoids repeatedly recomputing the long weather, calendar, and historical-feature lineage, 
which made earlier CV runs much slower and less stable.
"""

# Keep checkpointed intermediate results outside the notebook process.
spark.conf.set("spark.sql.debug.maxToStringFields", "50")
spark.catalog.clearCache()

CHECKPOINT_DIR = f"{hadoopFS}/user/{username}/assignment-2/checkpoints"
spark.sparkContext.setCheckpointDir(CHECKPOINT_DIR)

print(f"Checkpoint dir: {CHECKPOINT_DIR}")

# Materialize the final feature table once before cross-validation.
CV_INPUT_PATH = f"{hadoopFS}/user/{username}/assignment-2/part4_cv_input_parquet"

(
    feature_input_df
    .repartition("operating_day")
    .write
    .mode("overwrite")
    .parquet(CV_INPUT_PATH)
)

cv_input_df = spark.read.parquet(CV_INPUT_PATH)
print(f"Materialized CV input to: {CV_INPUT_PATH}")
print(f"Rows available for CV: {cv_input_df.count():,}")

temporal_folds = [
    # Winter: Dec-Jan-Feb
    ("Winter", "2024-03-01", "2025-02-28", "2025-12-01", "2026-02-28"),

    # Spring: Mar-Apr-May
    ("Spring", "2024-03-01", "2025-02-28", "2025-03-01", "2025-05-31"),

    # Summer: Jun-Jul-Aug
    ("Summer", "2024-06-01", "2025-05-31", "2025-06-01", "2025-08-31"),

    # Autumn: Sep-Oct-Nov
    ("Autumn", "2024-09-01", "2025-08-31", "2025-09-01", "2025-11-30"),
]

models = {
    "Linear Regression": LinearRegression(
        featuresCol="features",
        labelCol=label_col,
        maxIter=50,
        regParam=0.01,
        elasticNetParam=0.0,
        standardization=True,
        aggregationDepth=2,
    ),

    "Decision Tree": DecisionTreeRegressor(
        featuresCol="features",
        labelCol=label_col,
        maxDepth=10,
        minInstancesPerNode=20,
        seed=42,
    ),

    "Random Forest": RandomForestRegressor(
        featuresCol="features",
        labelCol=label_col,
        numTrees=30,
        maxDepth=8,
        minInstancesPerNode=20,
        seed=42,
    )
}


# %%
def evaluate_predictions(pred_df):
    """
    Compute regression and error-distribution metrics for a prediction DataFrame.
    """
    rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse").evaluate(pred_df)
    mae = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae").evaluate(pred_df)
    r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2").evaluate(pred_df)

    extra = (
        pred_df
        .withColumn("abs_error", F.abs(F.col("prediction") - F.col(label_col)))
        .withColumn("error", F.col("prediction") - F.col(label_col))
        .agg(
            F.expr("percentile_approx(abs_error, 0.90)").alias("p90_abs_error"),
            F.expr("percentile_approx(abs_error, 0.95)").alias("p95_abs_error"),
            F.avg("error").alias("bias"),
            F.avg(
                F.when(F.col("prediction") < F.col(label_col), 1.0)
                 .otherwise(0.0)
            ).alias("underprediction_rate")
        )
        .collect()[0]
    )

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "p90_abs_error": float(extra["p90_abs_error"]),
        "p95_abs_error": float(extra["p95_abs_error"]),
        "bias": float(extra["bias"]),
        "underprediction_rate": float(extra["underprediction_rate"]),
    }


def evaluate_mean_baseline(train_df, valid_df):
    """
    Evaluate a baseline that predicts the training-set mean delay for every row.
    """
    mean_delay = train_df.agg(F.avg(label_col).alias("mean_delay")).collect()[0]["mean_delay"]

    pred_df = valid_df.select(label_col).withColumn("prediction", F.lit(mean_delay))
    pred_df.cache()
    pred_df.count()

    metrics = evaluate_predictions(pred_df)
    pred_df.unpersist()

    return metrics


# %%
def run_temporal_cv(df, models, folds):
    """
    Run temporal cross-validation with fold-safe historical features and return
    one row of metrics per model and fold.
    """
    results = []

    for fold_name, train_start, train_end, valid_start, valid_end in folds:
        print(f"\n===== {fold_name} =====")
        print(f"Train period: {train_start} to {train_end}")
        print(f"Valid period: {valid_start} to {valid_end}")

        # DISK_ONLY avoids filling executor memory with huge cached DataFrames.
        train_raw_df = (
            df
            .filter(F.col("operating_day").between(train_start, train_end))
            .persist(StorageLevel.DISK_ONLY)
        )

        valid_raw_df = (
            df
            .filter(F.col("operating_day").between(valid_start, valid_end))
            .persist(StorageLevel.DISK_ONLY)
        )

        train_n = train_raw_df.count()
        valid_n = valid_raw_df.count()

        print(f"Train rows: {train_n:,}")
        print(f"Valid rows: {valid_n:,}")

        if train_n == 0 or valid_n == 0:
            print("Skipping fold because train or validation set is empty.")
            train_raw_df.unpersist(blocking=False)
            valid_raw_df.unpersist(blocking=False)
            continue

        # Fold-safe historical aggregate features.
        # Built from training fold only.
        start_hist = time.time()

        train_df, hist_numeric_cols = add_historical_delay_features(
            train_raw_df,
            train_raw_df,
            label_col=label_col
        )

        valid_df, _ = add_historical_delay_features(
            train_raw_df,
            valid_raw_df,
            label_col=label_col
        )

        fold_numeric_cols = numeric_cols + hist_numeric_cols

        train_df = prepare_ml_columns(
            train_df,
            categorical_cols,
            fold_numeric_cols,
            label_col
        ).checkpoint(eager=True)

        valid_df = prepare_ml_columns(
            valid_df,
            categorical_cols,
            fold_numeric_cols,
            label_col
        ).checkpoint(eager=True)

        hist_time = time.time() - start_hist

        print(f"Historical feature columns: {len(hist_numeric_cols)}")
        print(f"Total numeric columns:      {len(fold_numeric_cols)}")
        print(f"Historical feature time:    {hist_time:.2f}s")

        start_time = time.time()
        baseline_metrics = evaluate_mean_baseline(train_df, valid_df)
        baseline_time = time.time() - start_time

        results.append({
            "model": "Mean Baseline",
            "fold": fold_name,
            "train_period": f"{train_start} to {train_end}",
            "valid_period": f"{valid_start} to {valid_end}",
            "train_rows": int(train_n),
            "valid_rows": int(valid_n),
            "hist_feature_time_sec": float(hist_time),
            "train_time_sec": 0.0,
            "prediction_time_sec": float(baseline_time),
            **baseline_metrics
        })

        print(
            f"Mean Baseline | "
            f"RMSE={baseline_metrics['rmse']:.4f} | "
            f"MAE={baseline_metrics['mae']:.4f} | "
            f"R2={baseline_metrics['r2']:.4f}"
        )

        # Models
        for model_name, model in models.items():
            regression_type = "linear" if model_name == "Linear Regression" else "tree"

            pipeline = build_pipeline(
                model=model,
                categorical_cols=categorical_cols,
                numeric_cols=fold_numeric_cols,
                regression_type=regression_type
            )

            try:
                start_train = time.time()
                fitted_model = pipeline.fit(train_df)
                train_time = time.time() - start_train

                start_pred = time.time()
                pred_df = (
                    fitted_model
                    .transform(valid_df)
                    .select(label_col, "prediction")
                    .checkpoint(eager=True)
                )
                pred_time = time.time() - start_pred

                metrics = evaluate_predictions(pred_df)

                results.append({
                    "model": model_name,
                    "fold": fold_name,
                    "train_period": f"{train_start} to {train_end}",
                    "valid_period": f"{valid_start} to {valid_end}",
                    "train_rows": int(train_n),
                    "valid_rows": int(valid_n),
                    "hist_feature_time_sec": float(hist_time),
                    "train_time_sec": float(train_time),
                    "prediction_time_sec": float(pred_time),
                    **metrics
                })

                print(
                    f"{model_name} | "
                    f"RMSE={metrics['rmse']:.4f} | "
                    f"MAE={metrics['mae']:.4f} | "
                    f"R2={metrics['r2']:.4f} | "
                    f"train={train_time:.1f}s | "
                    f"pred={pred_time:.1f}s"
                )

                pred_df.unpersist(blocking=False)

            except Exception as e:
                print(f"ERROR while running {model_name} on {fold_name}: {e}")

        # Cleanup after each fold
        train_df.unpersist(blocking=False)
        valid_df.unpersist(blocking=False)
        train_raw_df.unpersist(blocking=False)
        valid_raw_df.unpersist(blocking=False)
        spark.catalog.clearCache()

    return spark.createDataFrame(pd.DataFrame(results))


# %% [markdown]
# > **Note:** Running the temporal cross-validation cell can take more than 3 hours on the cluster. If needed, this cell can be skipped; the resulting summary table is included in the markdown below.

# %%
cv_results_df = run_temporal_cv(cv_input_df, models, temporal_folds)

cv_results_df.orderBy("fold", "rmse").show(truncate=False)

# %%
# Save CV summary results so they can be reused later
CV_SUMMARY_PATH = f"{hadoopFS}/user/{username}/assignment-2/part4_cv_summary_v2"

# %% jupyter={"source_hidden": true}
cv_summary_df = (
    cv_results_df
    .groupBy("model")
    .agg(
        F.round(F.avg("rmse"), 4).alias("avg_rmse"),
        F.round(F.avg("mae"), 4).alias("avg_mae"),
        F.round(F.avg("r2"), 4).alias("avg_r2"),
        F.round(F.avg("p90_abs_error"), 4).alias("avg_p90_abs_error"),
        F.round(F.avg("p95_abs_error"), 4).alias("avg_p95_abs_error"),
        F.round(F.avg("bias"), 4).alias("avg_bias"),
        F.round(F.avg("underprediction_rate"), 4).alias("avg_underprediction_rate"),
        F.round(F.avg("train_time_sec"), 2).alias("avg_train_time_sec"),
        F.round(F.avg("prediction_time_sec"), 2).alias("avg_prediction_time_sec")
    )
    .orderBy("avg_rmse")
)

cv_summary_df.show(truncate=False)

(
    cv_summary_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(CV_SUMMARY_PATH)
)

print(f"Saved CV summary to: {CV_SUMMARY_PATH}")

# %%
saved_cv_summary_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(CV_SUMMARY_PATH)
)

# Display saved results
saved_cv_summary_df.show(truncate=False)

# %% [markdown]
# **Results**
#
# The temporal cross-validation results show that the Random Forest regressor performs best overall. It achieves the lowest average RMSE and MAE, and the highest average R² among the tested models. However, the improvement over Linear Regression is relatively small, while the training time is much higher.
#
# | Model | Avg RMSE | Avg MAE | Avg R² | Avg P90 abs. error | Avg P95 abs. error | Avg bias | Underprediction rate | Avg train time (s) | Avg prediction time (s) |
# |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | Random Forest | 3.1528 | 1.6544 | 0.0971 | 3.1120 | 4.7680 | -0.1616 | 0.3677 | 1358.72 | 8.10 |
# | Linear Regression | 3.1687 | 1.6787 | 0.0878 | 3.1239 | 4.6849 | -0.1189 | 0.3632 | 58.63 | 4.31 |
# | Decision Tree | 3.1856 | 1.6705 | 0.0779 | 3.2725 | 5.0213 | -0.1459 | 0.3708 | 377.13 | 5.26 |
# | Mean Baseline | 3.3244 | 1.7888 | -0.0040 | 2.8084 | 5.1626 | -0.1585 | 0.3343 | 0.00 | 10.14 |
#
# The Random Forest model is selected as the final model.

# %% [markdown]
# #### Feature-set experiments
#
# Before running the final temporal cross-validation, we tested several feature representations on smaller validation runs. These tests were used as feature-selection checks, while the temporal cross-validation below is used for the final model comparison.
#
# The first version included high-cardinality fields such as `trip_id` and `clean_arr_status`. This produced a very large sparse feature vector and did not improve the model; RMSE was slightly worse. Removing these fields made the feature vector much smaller and the training process faster.
#
# We also tested adding weather and calendar features. Weather features were kept because they reduced the validation error in the smoke tests, especially MAE. Calendar features were kept because they added planning-time signal through public holidays, school holidays, bridge days, and nearby holiday effects, although the improvement was smaller than for weather.
#
# We tested historical delay aggregates by stop, line, and time. These features improved RMSE and R², they had to be computed inside each temporal fold to avoid lookahead bias. Based on these experiments, the final pipeline uses the reduced planning-time categorical features, bus-specific weather features, calendar features, and fold-safe historical delay aggregates.

# %% [markdown]
# ### 6. Final model evaluation, interpretation and limitations

# %% [markdown]
# Our initial cross-validation showed that the random forest regressor was the top-performing model. Beyond our standard evaluations, we tested inference time and also evaluated the model's performance on the most severe delays. It is important that the model can process user queries efficiently. Also, the importance of severe delays is significantly higher than that of smaller ones; a 2' delay is not very important and perhaps acceptable from the commuters' standpoint, while a 10' delay may result in a missed connection.
#
# For the final parameter and runtime check, we train on 2025 because this is the most recent complete year available and the year the final model would be trained on. We evaluate on sampled 2024 records as a robustness check across a different year. Since the available 2024 data has a gap in August and September, those months are excluded from this check to avoid comparing against an incomplete period.

# %% [markdown]
# #### FInding the best parameters

# %%
# Prepare final train/test data for the selected model
base_df = feature_input_df

label_col = "label_delay_min"

categorical_cols = [
    "bpuic",
    "line_text",
    "hour",
    "day_of_week",
    "month",
]

numeric_cols = [c for c in weather_bus_cols if c in base_df.columns]
all_numeric_cols = weather_bus_cols  + calendar_numeric_cols
needed_cols = ["operating_day", label_col] + categorical_cols + all_numeric_cols

missing_cols = [c for c in needed_cols if c not in base_df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in feature_input_df: {missing_cols}")

raw_df = (
    base_df
    .select(*needed_cols)
    .filter(F.col("operating_day").isNotNull())
    .filter(F.col(label_col).isNotNull())
    .withColumn("operating_day_int", F.unix_date(F.col("operating_day")))
)

# Clean categorical columns but keep them usable for joins
for c in categorical_cols:
    raw_df = raw_df.withColumn(c, F.coalesce(F.col(c).cast("string"), F.lit("UNKNOWN")))

for c in numeric_cols:
    raw_df = raw_df.withColumn(c, F.coalesce(F.col(c).cast("double"), F.lit(0.0)))

raw_df = raw_df.withColumn(label_col, F.col(label_col).cast("double")).cache()

total_rows = raw_df.count()

# Temporal split: first 80% time train, last 20% time test
split_day_int = raw_df.approxQuantile("operating_day_int", [0.8], 0.01)[0]

train_raw_df = (
            raw_df
            .filter(F.col("operating_day").between("2025-01-01", "2025-12-31"))
            .persist(StorageLevel.DISK_ONLY)
        )

test_raw_df = (
            raw_df
            .filter(F.col("operating_day").between("2024-01-01", "2024-12-31"))
            .filter(~F.col("operating_day").between("2024-08-01", "2024-09-30")) # Since we do not currently have data for 
            .persist(StorageLevel.DISK_ONLY)
            .sample(0.01, seed=490)
        )

train_enriched_df, hist_numeric_cols = add_historical_delay_features(
    train_raw_df,
    train_raw_df,
    label_col=label_col
)

test_enriched_df, _ = add_historical_delay_features(
    train_raw_df,
    test_raw_df,
    label_col=label_col
)

for c in all_numeric_cols:
    train_enriched_df = train_enriched_df.withColumn(c, F.coalesce(F.col(c).cast("double"), F.lit(0.0))).cache()
    testing_enriched_df = test_enriched_df.withColumn(c, F.coalesce(F.col(c).cast("double"), F.lit(0.0))).cache()

index_cols = [c + "_index" for c in categorical_cols]
encoded_cols = [c + "_ohe" for c in categorical_cols]


# %%
def grid_search(pipeline, grid_trees, grid_depth, train_df, valid_df):
    "Performs grid search using for selected parameter values of the model."
    rmse = RegressionEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="rmse"
    )
    res = np.empty((len(grid_trees), len(grid_depth)))
    regressor = pipeline.getStages()[-1]
    for i, tree in enumerate(grid_trees):
        for j, depth in enumerate(grid_depth):
            
            start = time.time()
            regressor.setParams(numTrees=tree, maxDepth=depth)
            model = pipeline.fit(train_df)
            end = time.time()
            print((f"Training time: {end-start}"))
            
            pred = model.transform(valid_df)
            err = rmse.evaluate(pred)
            print(f"Number of trees: {tree}, maximum depth: {depth}, RMSE:{err}.")
            res[i, j] = err
            
    best_i, best_j = np.unravel_index(np.argmin(res), res.shape)
    best_trees, best_depth = grid_trees[best_i], grid_trees[best_j]
    print(f"Best combination of parameters: numTrees={best_trees}, maxDepth={best_depth}.")
    return best_trees, best_depth


# %%
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol=label_col,
    minInstancesPerNode=20,
    seed=42,
)

final_pipeline = build_pipeline(
    model=rf,
    categorical_cols=categorical_cols,
    numeric_cols=all_numeric_cols,
    regression_type="tree",
)

trees = [20, 30]
depths = [5, 8]

val_enriched_df, test_enriched_df = (
    test_enriched_df
    .filter(F.col("operating_day").between("2024-01-01", "2024-12-31"))
    .filter(~F.col("operating_day").between("2024-08-01", "2024-09-30"))
    .sample(0.1, seed=42)
    .persist(StorageLevel.DISK_ONLY)
    .randomSplit([0.5, 0.5], seed=42)
)

# %% [markdown]
# > **Note:** The Random Forest grid search takes more than 2 hours on the cluster. We ran it once and saved the result screenshot in the `results/` folder. The best parameters found were `numTrees=20` and `maxDepth=5`, so the notebook below directly assigns these values instead of rerunning the grid search.

# %%
# Skip if you dont want to wait
best_trees, best_depth = grid_search(final_pipeline, trees, depths, train_enriched_df, val_enriched_df)

# %%
# Grid search was run once; reuse the best parameters to keep the notebook runnable.
best_trees = 20
best_depth = 5

# %%

# %% [markdown]
# #### Training the final model

# %% [markdown]
# The final Random Forest model is trained on last year data from 2025-01-01 to 2025-12-31, using the selected parameters numTrees=20 and maxDepth=5.

# %%
# Model training takes around 15 minutes
final_regressor = final_pipeline.getStages()[-1]
final_regressor.setParams(numTrees=best_trees, maxDepth=best_depth)

model = final_pipeline.fit(train_enriched_df)


# %%
def evaluate_runtime(model, data):
    """
    Measure how long the fitted model takes to generate predictions.
    The count on pred forces Spark to execute the transformation.
    """
    start = time.time()
    pred = model.transform(data)
    n_rows = pred.count()
    end = time.time()

    print(f"Evaluated {n_rows:,} data points in {end - start:.2f} seconds.")
    return end - start


runtime = evaluate_runtime(model, data=test_enriched_df)
print(runtime)


# %% [markdown]
# We will now evaluate the model's performance for signficant delays. We will compare the RMSE and MAE of our model with the empirical minimzers, i.e. the mean and median of the dataset, respectively.

# %%
def evaluate_large_delays(model, data):
    """
    Evaluate RMSE and MAE on large delays, defined as delays at or above
    the 90th percentile of the evaluation data.
    """
    
    p50, p90 = data.approxQuantile('label_delay_min', [0.5, 0.9], relativeError=0.05)
    mean = data.select(F.mean(data.label_delay_min)).first()[0]
    
    data_p90 = data.filter(data.label_delay_min>=p90)
    pred = model.transform(data_p90)
    
    rmse = RegressionEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="rmse"
    )
    rmse_pred = rmse.evaluate(pred)
    
    mae = RegressionEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="mae"
    )

    mae_pred = mae.evaluate(pred)
    
    print(f"RMSE of empirical mean: {data_p90.select(F.sqrt(F.mean((data_p90.label_delay_min-mean)**2))).first()[0]}.")
    print(f"Model RMSE: {rmse_pred}.")
    print(f"MAE of empirical median: {data_p90.select(F.mean(F.abs(data_p90.label_delay_min-p50))).first()[0]}.")
    print(f"Model MAE: {mae_pred}.")
    return rmse_pred, mae_pred

evaluate_large_delays(model, test_enriched_df)

# %% [markdown]
# While the model struggles with capturing precise delays across the entire testing data set, it is more capable at predicting large delays, which is an important aspect. With that in mind, we believe the model could be useful in providing delay estimates with an acceptable inference time. 

# %% [markdown]
# Overall, the final evaluation shows that the model learns meaningful delay patterns despite the noisy nature of public transport data. The Random Forest model improves over the mean baseline in RMSE, MAE, and R², which suggests that the selected features provide real predictive signal.
#
# For the robust journey-planning use case, this is a valuable result. The model does not need to predict every delay perfectly; it needs to provide better delay-risk estimates than a naive baseline. These estimates can help the planner avoid fragile connections and prefer routes with safer transfer margins.

# %% [markdown]
# #### Limitations

# %% [markdown]
# Since the model has been trained on weather data, it is expected to perform worse on future data when the actual weather is unknown. The performance of the algorithm is therefore dependent on the accuracy of the weather forecast; delays will be easier to predict in the near future and (significantly) harder in the far future. Per the [United States National Environmental Satellite, Data, and Information Service](https://www.nesdis.noaa.gov/about/k-12-education/weather-forecasting/how-reliable-are-weather-forecasts), a 10-day forecast is only right ~50% of the time. In order for the algorithm to be used in the future, historical weather data can be passed, either as mean/median values or Confidence Intervals. For example, the precipitation probability can be used as the `is_rainy_bus` value, despite it not being a binary. 
#
# The model is also limited by the features available at planning time. It does not know about live disruptions, accidents, vehicle breakdowns, or sudden operational changes unless those are added as extra real-time inputs. Therefore, the model should be interpreted as a historical and contextual delay estimator, not as a complete real-time disruption detector.
#
# Future work could incorporate live delay observations, disruption feeds, and traffic data. Traffic data would be especially useful for bus services, where road congestion can strongly affect arrival delays. Another useful extension would be adding uncertainty estimates, such as prediction intervals, so the journey planner can reason not only about expected delay but also about the risk of missing a connection.

# %% [markdown]
# # That's all, folks!
#
# Be nice to other, do not forget to close your spark session.

# %%
# Yes we made sure to be nice to others :) 
spark.stop()
