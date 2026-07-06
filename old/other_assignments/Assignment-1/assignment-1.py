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
# # DSLab Assignment 1
# ---
#
# In this assignment, you will use the SBB timetable data to model the public transport infrastructure in the Lausanne region.
#
# Your objective is to design a method for building a comprehensive representation of the public transport network and to implement routing algorithms that operate on this network. This work will form the foundation of your final project.
#
# You are encouraged to choose the data structures and algorithms that best suit your intended final implementation. The steps outlined in this assignment are meant to provide a solid starting point, but you are free to adapt them if you find more suitable approaches.
#
# To achieve this, you will work with the SBB timetable data, including:
#
# * stop locations,
# * information about journeys and trips,
# * scheduled arrival and departure times at each stop.
#
# The assignment is in two parts:
#
# * Using this information, you will build a data model of the transport network that can be used to compute routes between stops within the designated region.
# * You will then validate your method by creating an interactive visualization that displays travel times from a specified stop to other stops in the region.
#
# ⚠️ Important notes (please read carefully)
#
# * The routing algorithm question is open-ended. Points will be awarded based on the quality of both the proposed algorithm and its implementation. You will receive credit for proposing a well-designed algorithm (which you must describe), even if the implementation is not fully completed.
# * It is not mandatory to use the large-scale platform for every part of this assignment. For lightweight tasks—such as implementing the routing algorithm—plain Python may be sufficient, especially if you reduce the input data size. You may use any Python packages you find useful, but please contact us so we can install them in the shared environment if needed.
# * Points **will** be deducted if the notebook consistently breaks and cannot be executed (for example, due to undefined Python variables).
# * The same code must run for all members of your group. Points may be deducted if the notebook fails for some users, for instance due to hard-coded user parameters or paths that cause permission errors.
# * The performance of your solution will be evaluated along two dimensions:
#
#   - Data preparation time: the one-time cost required to prepare the data used by your algorithm (e.g., creating tables, preprocessing, building indexes or intermediate structures).
#   - Routing time: the time required by your routing algorithm to compute routes once the data has been prepared.
#   - Your performance will be evaluated relative to the class results for both components. Implementations that are **significantly slower** than the typical range (either in preprocessing time or routing time) will receive lower scores.
#
# ## Hand-in Instructions
#
# - __Due: 31.03.2026 23h59 CET__
# - Create a fork of this repository **under your group gitlab repo**.
#     - In a browser, navigate to the [assignment](https://dslabgit.datascience.ch/course/2026/assignment-1)
#     - Click _Fork_ (Create New Fork)
#     - In the Project URL, select your group namespace, e.g. "students/2026/Z9", **do not** change the project slug !
# - Set the python groupName variable below, e.g. _groupName='Z9'_
# - Do not change the name of this file.
# - `git push` your final verion to the master branch of your **group's repository** before the due date.
# - Add necessary comments and discussion to make your codes readable.
# - Let us know if you need us to install additional python packages.
#
# ## Useful references
#
# In this part you will make good use of DQL statements of nested SELECT, GROUP BY, JOIN, IN, DISTINCT, and Geo Spatial UDF.
#
# * [Trino documentation](https://trinodb.github.io/docs.trino.io/479/sql.html)

# %% [markdown]
# <div style="font-size: 100%" class="alert alert-block alert-warning">
#     <b>Fair cluster Usage:</b>
#     <br>
#     As there are many of you working with the cluster, we encourage you to prototype your queries on small data samples before running them on whole datasets. Do not hesitate to partion your tables, and LIMIT the output of your queries to a few rows to begin with. You are also free to test your queries using alternative solutions such as <i>DuckDB</i>.
#     <br><br>
#     You may lose your session if you remain idle for too long or if you interrupt a query. If that happens you will not lose your tables, but you may need to reconnect to the data query engine.
#     <br><br>
# </div>

# %% [markdown]
# ---
# **Note**: all the data used in this homework is described in the [final preview](./final-preview.md) document, which can be found in this repository. The document is a preview the final project due for the end of this semester, so that you can put this assignment into the broader context.
#
# For this notebook you are free to use the following tables, which can all be found under the _iceberg.com490_iceberg_ namespace shared by the class (you may use the _sharedns_ variable).
# - You can list the tables with the command `f"SHOW TABLES IN {sharedns}"`.
# - You can see the details of each table with the command `f"DESCRIBE {sharedns}.{table_name}"`.
#
# ---
# For your convenience we also define useful python variables:
#
# * _hadoopfs_
#     * The HDFS server, in case you need it for hdfs, pandas or pyarrow commands.
# * _username_:
#     * Your user id (EPFL gaspar id), use it as your personal namespace for your private tables.
# * _sharedns_:
#     * This is an alias for the Trino schema _iceberg.com490_iceberg_, it is read only for you.
# * _userns_:
#     * Your personal Trino schema, store your personal tables there.
# * _grouphdfs_:
#     * hdfs folder of your group
#  
# If you would like to share a Trino table schema with other people from your group, please contact us.

# %%
groupName='J1' # TODO set your group name here.

# %%
import os
import warnings
import base64 as b64
import json
import time
import re

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable .*")

def getUsername():
    payload = os.environ.get('EPFL_COM490_TOKEN').split('.')[1]
    payload=payload+'=' * (4 - len(payload) % 4)
    obj = json.loads(b64.urlsafe_b64decode(payload))
    if (time.time() > int(obj.get('exp')) - 3600):
        raise Exception('Your credentials have expired, please restart your Jupyter Hub server:'
                        'File>Hub Control Panel, Stop My Server, Start My Server.')
    time_left = int((obj.get('exp') - time.time())/3600)
    return obj.get('sub'), time_left


# %%
username, validity_h = getUsername()

hadoopfs = os.environ.get('HADOOP_FS')
userns   = 'iceberg.' + username + '_iceberg'
sharedns = 'iceberg.com490_iceberg'
groupfs  = f"{hadoopfs}/user/groups/com-490/{groupName}"

if not re.search('([A-Z][0-9Z])', groupName):
    raise Exception(f"Invalid group name {groupName}")

print(f"You are: {username} of group {groupName}")
print(f"credentials validity: {validity_h} hours left.")
print(f"personal namespace:   {userns}")
print(f"Group HDFS folder:    {groupfs}")


# %% [markdown]
# ---

# %%
import pandas as pd
import trino
from contextlib import closing
from urllib.parse import urlparse
from trino.dbapi import connect
from trino.auth import BasicAuthentication, JWTAuthentication

trinoAuth = JWTAuthentication(os.environ.get('EPFL_COM490_TOKEN'))
trinoUrl  = urlparse(os.environ.get('TRINO_URL'))
Query=[]

print(f"Connecting to Data Query Engine URL: {trinoUrl.scheme}://{trinoUrl.hostname}:{trinoUrl.port}/")

conn = connect(
    host=trinoUrl.hostname,
    port=trinoUrl.port,
    auth=trinoAuth,
    http_scheme=trinoUrl.scheme,
    verify=True
)

print('You are connected!')

# %% [markdown]
# ## Part I. Prepare your environment and explore - 10 Points

# %% [markdown]
# Declare a "managed" Trino schema under your username, called {userns}. This is the schema under which you will define the tables, views or materialized views required to model your infrastructure. Managed means you do not specify a default location for your schema and allow the system to manage it for you.

# %%
# %%time
# TODO Create the schema
with closing(conn.cursor()) as cur:
    cur.execute(f"""CREATE SCHEMA IF NOT EXISTS {userns}""")

# %%
query = f"""
SELECT uuid, name, region, level
FROM {sharedns}.geo
WHERE name IN ('Lausanne', 'Ouest lausannois')
"""
pd.read_sql(query, conn)

# %% [markdown]
# ### Explore the SBB data
#
# Use the Data Query Engine to explore the timetable tables, in particular _{sharedns}.sbb_stops_ and _{sharedns}.sbb_stop_times_.
#
# * Find the most recent publication date, and restrict your analysis in this notebook to the timetables corresponding to that date.
#
# * Identify the field(s) used across the tables to represent stop locations.
#
# * Determine how many distinct stop locations exist in Switzerland, ignoring platform-level information (see the SBB documentation on the stop ID naming structure, or infer it from the data).
#
# * Analyze the value ranges, format patterns, null values, and potential invalid entries in the tables. In particular, examine the time fields used in the schedules, paying attention to their ranges and to values appearing at the start or end of a trip (for example, by inspecting a specific trip).
#
# You will use in part II the insights gained from this exploration to implement the data transformations needed to construct the public transport topology used in your data model.
#
# Note: Do not include every query you tried. Only present the queries that are most relevant for answering the questions above or that help justify your design choices. Whenever possible, limit their output to the minimum number of rows necessary to support your observations.
#
# You must answer at a minimum the questions listed below, but feel free to explore more.

# %% [markdown]
# a) Most recent publication date
#
# In all the remaining questions you must only query timetables corresponding to that date.

# %%
# %%time
## TODO explore the table and find max_pub_date, the date of publication of the most recent time tables.

max_date = f"""
    SELECT MAX(pub_date) AS max_pub_date 
    FROM {sharedns}.sbb_stops
"""

df_max_date = pd.read_sql(max_date, conn)
max_pub_date = df_max_date.iloc[0]['max_pub_date']

print(f"The most recent publication date is: {max_pub_date}")
# %%

# %%


# %% [markdown]
# b) How many distinct stops are in Switzerland ? If you have multiple platforms at a stop, count them as a single stop.
#
# See the SBB [gtfs](https://opentransportdata.swiss/en/cookbook/timetable-cookbook/gtfs/) stop data, and note that we conform to the schema from before June 2025.

# %%
# %%time

distinct_stops = f"""
    SELECT COUNT(DISTINCT split_part(stop_id, ':', 1)) AS distinct_swiss_stops
    FROM {sharedns}.sbb_stops
    WHERE pub_date = DATE '{max_pub_date}'
      AND split_part(stop_id, ':', 1) LIKE '85%'
"""

df_distinct_stops = pd.read_sql(distinct_stops, conn)
swiss_stops_count = df_distinct_stops.iloc[0]['distinct_swiss_stops']

print(f"There are {swiss_stops_count} distinct stops in Switzerland.")
# %%

# %%


# %% [markdown]
# ---

# %% [markdown]
# c) Observe the arrival and departure times (hour of day) of the sbb stop times table, compute their range, and share your results.

# %%
# %%time

time_ranges = f"""
    SELECT 
        MIN(arrival_time) AS min_arrival,
        MAX(arrival_time) AS max_arrival,
        MIN(departure_time) AS min_departure,
        MAX(departure_time) AS max_departure
    FROM {sharedns}.sbb_stop_times
    WHERE pub_date = DATE '{max_pub_date}'
"""

df_time_ranges = pd.read_sql(time_ranges, conn)

min_arr = df_time_ranges.iloc[0]['min_arrival']
max_arr = df_time_ranges.iloc[0]['max_arrival']
min_dep = df_time_ranges.iloc[0]['min_departure']
max_dep = df_time_ranges.iloc[0]['max_departure']

print("--- SBB Stop Times Range ---")
print(f"Arrival Time Range:   {min_arr} to {max_arr}")
print(f"Departure Time Range: {min_dep} to {max_dep}")

# %% [markdown]
# **Observation on Time Ranges:**
# The arrival and departure times extend well beyond the standard 24-hour clock, reaching up to 46:24:00. This is not a data error. Because the SBB schedules use the GTFS format, an arrival or departure time exceeding 24 hours indicates that the transport completes its journey on the days following the start of the operational day. For example, 25:30:00 means 01:30 AM on the next day.

# %%
# %%time
# Count how many stop times occur after the 24-hour mark

midnight_stats = f"""
    SELECT
        COUNT(*) AS total_stop_times,
        COUNT(*) FILTER (WHERE arrival_time >= '24:00:00') AS stops_after_midnight
    FROM {sharedns}.sbb_stop_times
    WHERE pub_date = DATE '{max_pub_date}'
"""

df_midnight_stats = pd.read_sql(midnight_stats, conn)
total = df_midnight_stats.iloc[0]['total_stop_times']
after_mid = df_midnight_stats.iloc[0]['stops_after_midnight']
percentage = (after_mid / total) * 100

print(f"Total stop times: {total}")
print(f"Stops after midnight (>24h): {after_mid} ({percentage:.2f}%)")

# %% [markdown]
# **Observation:** nearly 3.76% of all stop times occur after 24:00:00.

# %%
# %%time

# Look at specific trips that run past 46 hours
extreme_trips = f"""
    SELECT 
        st.trip_id, 
        st.stop_id, 
        s.stop_name, 
        st.stop_sequence, 
        st.arrival_time, 
        st.departure_time
    FROM {sharedns}.sbb_stop_times st
    JOIN {sharedns}.sbb_stops s 
      ON st.stop_id = s.stop_id
    WHERE st.pub_date = DATE '{max_pub_date}'
      AND s.pub_date = DATE '{max_pub_date}'
      AND st.arrival_time >= '46:00:00'
    LIMIT 5
"""

df_extreme_trips = pd.read_sql(extreme_trips, conn)
display(df_extreme_trips)


# %% [markdown]
# **Observation on Extreme Time Values:**
# In this dataset, when a stop is the start or end of a journey, the `arrival_time` is typically identical to the `departure_time`. Furthermore, times exceeding 24 hours represent journeys rolling over into the next day. However, an extreme value like 46:24:00 warrants deeper investigation to determine if it represents a genuine long-distance international route or a potential invalid entry.

# %%
# %%time

# Investigate the previous trip: 56.TA.91-13-F-j26-1.37.R

specific_trip = f"""
    SELECT 
        st.stop_sequence, 
        s.stop_name, 
        substr(st.stop_id, 1, 2) AS country_code,
        st.arrival_time, 
        st.departure_time
    FROM {sharedns}.sbb_stop_times st
    JOIN {sharedns}.sbb_stops s 
      ON st.stop_id = s.stop_id
    WHERE st.pub_date = DATE '{max_pub_date}'
      AND s.pub_date = DATE '{max_pub_date}'
      AND st.trip_id = '56.TA.91-13-F-j26-1.37.R'
    ORDER BY st.stop_sequence
"""

df_specific_trip = pd.read_sql(specific_trip, conn)
display(df_specific_trip)

# %% [markdown]
# **Conclusion on Potential Invalid Entries:**
# By tracking the full sequence of the extreme trip (`56.TA.91-13-F-j26-1.37.R`), we discovered it is not a 46-hour international journey, but a short regional route entirely within Switzerland (country code `85`). The train departs Sevelen at 22:20:00 and supposedly arrives at the neighboring station of Sargans at 46:24:00. 
#
# Subtracting exactly 24 hours from 46:24:00 yields 22:24:00, which represents a highly realistic 4-minute travel time. This confirms that the 46-hour timestamp is a **data anomaly** (a +24-hour shift error) rather than a legitimate schedule. Identifying such anomalies is critical, as they must be handled gracefully when implementing the routing algorithm later to avoid suggesting routes that take a full extra day.

# %% [markdown]
# d) How many distinct trips are serving at least one stop in switzerland ?

# %%
# %%time

swiss_trips = f"""
    SELECT COUNT(DISTINCT trip_id) AS distinct_swiss_trips
    FROM {sharedns}.sbb_stop_times
    WHERE pub_date = DATE '{max_pub_date}'
      AND split_part(stop_id, ':', 1) LIKE '85%'
"""

df_swiss_trips = pd.read_sql(swiss_trips, conn)
swiss_trips_count = df_swiss_trips.iloc[0]['distinct_swiss_trips']

print(f"There are {swiss_trips_count} distinct trips serving at least one stop in Switzerland.")
# %%


# %% [markdown]
# e) Among the stops in Switzerland counted in b) how many distinct stops appear in {sharedns}.sbb_istdaten (bpuic)

# %%
# %%time

istdaten_stops = f"""
    SELECT COUNT(DISTINCT bpuic) AS matching_stops
    FROM {sharedns}.sbb_istdaten
    WHERE bpuic IN (
        SELECT CAST(split_part(stop_id, ':', 1) AS INTEGER)
        FROM {sharedns}.sbb_stops
        WHERE pub_date = DATE '{max_pub_date}'
          AND split_part(stop_id, ':', 1) LIKE '85%'
    )
"""

df_istdaten_stops = pd.read_sql(istdaten_stops, conn)
matching_stops_count = df_istdaten_stops.iloc[0]['matching_stops']

print(f"Out of the {swiss_stops_count} scheduled Swiss stops, {matching_stops_count} appear in the actual data (istdaten).")
# %%

# %%


# %% [markdown]
# ---

# %% [markdown]
# ## Part II. 50 Points

# %% [markdown]
# In this second part, you will build a comprehensive data representation of the public transport network, forming the foundation for your final project.
#
# To keep code execution manageable, you should limit your analysis to the _Lausanne_ and Ouest _Lausannois districts_, whose shapes are available in the _{shared}.geo_ table.
#
# However, your methods must remain region-agnostic, so that the notebook can be applied to other regions without modification. Ideally, your implementation should be configurable, allowing you to define a list of regions that your algorithm covers.

# %% [markdown]
# ### a) Find all the stops in region of interest - 5/50
#
#

# %% [markdown]
# * Explore _{sharedns}.geo_ (from _swiss topo_) and find the records containing the _wkb_geometry_ shapes of the _Lausanne_ and _Ouest lausannois_ districts.
# * Find all the stops in the above districts from _{sharedns}.sbb_stops_ (use [geo spatial](https://trino.io/docs/471/functions/geospatial.html) functions)
# * Save the results into a materialized view _{userns}.stops_ using the CTAS (Create Table As Select) or CREATE + INSERT approach.
# * The resulting view _{userns}.stops_ is a subset of table _{sharedNS}.sbb_stops_:
#     * _stop_id_
#     * _stop_name_
#     * _stop_lat_
#     * _stop_lon_
#
# Platforms belonging to the same stop can be merged into a single stop, with the stop’s location defined as the average of the platform coordinates. "Parent" stops are cluster of stops and can be ignored. All remaining stop Ids should be numerical.

# %%
# %%time
# TODO find the stops and create a materialized view
# 1. Explore the geo table for the two districts
explore_geo = f"""
    SELECT uuid, name, region, level
    FROM {sharedns}.geo
    WHERE name IN ('Lausanne', 'Ouest lausannois')
"""

df_geo = pd.read_sql(explore_geo, conn)
display(df_geo)

# %% [markdown]
# **Observation:**
# 2 types of levels (city / distirict), since the question is Lausanne and Ouest Lausannois **districts** we will add a filter `level = 'district'`

# %%
# %%time
# Explore the distribution of location_type for Swiss stops
explore_location_types = f"""
    SELECT 
        location_type, 
        COUNT(*) AS stop_count
    FROM {sharedns}.sbb_stops
    WHERE pub_date = DATE '{max_pub_date}'
      AND split_part(stop_id, ':', 1) LIKE '85%'
    GROUP BY location_type
    ORDER BY stop_count DESC
"""

df_location_types = pd.read_sql(explore_location_types, conn)
display(df_location_types)

# %%
# %%time
# 2. Find all the stops in the above districts using geospatial functions
find_stops_preview = f"""
    WITH target_regions AS (
        SELECT wkb_geometry
        FROM {sharedns}.geo
        WHERE name IN ('Lausanne', 'Ouest lausannois')
          AND level = 'district'
    ),
    merged_stops AS (
        SELECT 
            CAST(split_part(stop_id, ':', 1) AS INTEGER) AS stop_id,
            MAX(stop_name) AS stop_name,
            AVG(stop_lat) AS stop_lat,
            AVG(stop_lon) AS stop_lon
        FROM {sharedns}.sbb_stops
        WHERE pub_date = DATE '{max_pub_date}'
          AND split_part(stop_id, ':', 1) LIKE '85%'
        -- We removed the location_type filter because we proved it is always NULL!
        GROUP BY CAST(split_part(stop_id, ':', 1) AS INTEGER)
    )
    SELECT DISTINCT s.stop_id, s.stop_name, s.stop_lat, s.stop_lon
    FROM merged_stops s
    JOIN target_regions r 
      ON ST_Contains(ST_GeomFromBinary(r.wkb_geometry), ST_Point(s.stop_lon, s.stop_lat))
"""

df_stops_preview = pd.read_sql(find_stops_preview, conn)
print(f"Found {len(df_stops_preview)} distinct stops in the Lausanne and Ouest lausannois districts.")
display(df_stops_preview.head(10))

# %%
# %%time
# 3. Save the results into a materialized view/table

# Drop it first so you can safely rerun this cell if needed
drop_stops = f"DROP MATERIALIZED VIEW IF EXISTS {userns}.stops"
with closing(conn.cursor()) as cur:
    cur.execute(drop_stops)

# Execute the CTAS command, It's mostly a copy paste from the previous cell because It's better to be sure of what we are saving to the DB.
create_stops_table = f"""
    CREATE MATERIALIZED VIEW {userns}.stops AS
    WITH target_regions AS (
        SELECT wkb_geometry
        FROM {sharedns}.geo
        WHERE name IN ('Lausanne', 'Ouest lausannois')
          AND level = 'district'
    ),
    merged_stops AS (
        SELECT 
            CAST(split_part(stop_id, ':', 1) AS INTEGER) AS stop_id,
            MAX(stop_name) AS stop_name,
            AVG(stop_lat) AS stop_lat,
            AVG(stop_lon) AS stop_lon
        FROM {sharedns}.sbb_stops
        WHERE pub_date = DATE '{max_pub_date}'
          AND split_part(stop_id, ':', 1) LIKE '85%'
        GROUP BY CAST(split_part(stop_id, ':', 1) AS INTEGER)
    )
    SELECT DISTINCT s.stop_id, s.stop_name, s.stop_lat, s.stop_lon
    FROM merged_stops s
    JOIN target_regions r 
      ON ST_Contains(ST_GeomFromBinary(r.wkb_geometry), ST_Point(s.stop_lon, s.stop_lat))
"""

with closing(conn.cursor()) as cur:
    cur.execute(create_stops_table)

# Let's dynamically count the rows to prove it worked!
count_stops = f"SELECT count(*) AS total_stops FROM {userns}.stops"
actual_count = pd.read_sql(count_stops, conn).iloc[0]['total_stops']
print(f"Success! Table {userns}.stops has been created with {actual_count} rows.")
# %%

# %%

# %%


# %% [markdown]
# ### b) Find stops pairs within walking distance of each other - 5/50
#
# Use the results of table {userns}.stops to identify all pairs of stops that are within 500 m of each other, and save the results in the materialized view _{userns}.stop_to_stop_.
#
# At a minimum, the view should include:
#
# - a_stop_id
# - b_stop_id
# - distance: straight-line distance in meters from a_stop_id to b_stop_id
#
# It is your choice whether it should represent directed or undirected stop pairs, as long as your routing algorithm is able to handle it.

# %% [markdown]
# **Critical Design Choice:** (How should we store the pairs?)
#
# - **Undirected:** `(S1, S2)` represents `S1 <-> S2`
#   - **Advantages:**
#     1. **Saves Memory and Storage:** We only need to store half the number of edges.
#     2. **Simpler Logic:** Standard algorithms are often easier to implement and debug.
#     3. **Fits the Walking Assumption:** The project instructions explicitly state we can calculate walking distances as a "straight line" (As the Crow Flies). Because of this rule, walking from S1 to S2 is mathematically identical to walking from S2 to S1.
#
# - **Directed:** `(S1, S2)` represents `S1 -> S2` (and we must also store `S2 -> S1`)
#   - **Advantages:**
#     1. **Handles Asymmetric Reality:** While *walking* is symmetric in this model, actual transit is not. A bus line often takes a different route or uses different stops on its return journey. (MBC 705 Bus Example)
#     2. **Accounts for Delays:** Traffic or historical delays often impact one direction of travel heavily while the other remains fine.
#     3. **Matches the Transit Data:** Since train and bus trips are inherently directed, making our walking edges directed too means the routing algorithm only has to handle one unified type of logic.
#  
# **Choice:** Directed Pairs.

# %%
# %%time
# TODO create a materialized view of stop pairs

# Drop the table if it already exists
drop_stop_pairs = f"DROP MATERIALIZED VIEW IF EXISTS {userns}.stop_to_stop"
with closing(conn.cursor()) as cur:
    cur.execute(drop_stop_pairs)

# 2. Execute the CTAS command with extra features(see below)
create_stop_pairs = f"""
    CREATE MATERIALIZED VIEW {userns}.stop_to_stop AS
    WITH calculated_distances AS (
        SELECT 
            a.stop_id AS a_stop_id,
            a.stop_name AS a_stop_name,
            a.stop_lat AS a_stop_lat,
            a.stop_lon AS a_stop_lon,
            b.stop_id AS b_stop_id,
            b.stop_name AS b_stop_name,
            b.stop_lat AS b_stop_lat,
            b.stop_lon AS b_stop_lon,
            -- great_circle_distance returns kilometers, multiply by 1000 for meters
            great_circle_distance(a.stop_lat, a.stop_lon, b.stop_lat, b.stop_lon) * 1000 AS distance
        FROM {userns}.stops a
        CROSS JOIN {userns}.stops b
        WHERE a.stop_id != b.stop_id -- Directed edges (A->B and B->A), no self-loops
    )
    SELECT 
        a_stop_id,
        b_stop_id,
        a_stop_name,
        b_stop_name,
        a_stop_lat,
        a_stop_lon,
        b_stop_lat,
        b_stop_lon,
        distance,
        -- Apply the formula: 2 min base transfer + 1 min per 50m
        2.0 + (distance / 50.0) AS walk_time_min 
    FROM calculated_distances
    WHERE distance <= 500
"""

with closing(conn.cursor()) as cur:
    cur.execute(create_stop_pairs)

row_count = pd.read_sql(f"SELECT COUNT(*) FROM {userns}.stop_to_stop", conn).iloc[0, 0]

print(f"Success! Table {userns}.stop_to_stop has been created with {row_count} rows.")
check_pairs = f"""
    SELECT a_stop_name, b_stop_name, distance 
    FROM {userns}.stop_to_stop 
    LIMIT 5
"""
display(pd.read_sql(check_pairs, conn))

# %% [markdown]
# ### **Hand-off Note for Part II (c) & (d)**
#
# **Note to Team:** While the assignment instructions for **Part II (b)** only required `a_stop_id`, `b_stop_id`, and `distance`, I have taken the initiative to include several additional features in the `{userns}.stop_to_stop` table to simplify our next steps: (feel free to Update)
#
# * **Stop Names & Geospatial Coordinates:** Included for both stops to allow for immediate plotting in **Part II (c)** without extra joins.
# * **Walking Time (`walk_time_min`):** Pre-calculated using the official SBB transfer formula (2 min base + 1 min per 50m) to serve as edge weights for our routing algorithm.

# %%
# %%time
# Displaying all features in the stop_to_stop table as requested
check_all_features = f"""
    SELECT 
        a_stop_id,
        b_stop_id,
        a_stop_name,
        b_stop_name,
        a_stop_lat,
        a_stop_lon,
        b_stop_lat,
        b_stop_lon,
        distance,
        walk_time_min
    FROM {userns}.stop_to_stop
    LIMIT 10
"""

df_all_features = pd.read_sql(check_all_features, conn)
display(df_all_features)

print(f"Total walking edges available: {len(pd.read_sql(f'SELECT * FROM {userns}.stop_to_stop', conn))}")

# %% [markdown]
# ### c) Display selected stops - 5/50
#
# * Use plotly or similar plot framework to display all the stop locations in Lausanne region on a map (scatter plot or heatmap).
# * Display straight lines connecting stops that are within 250 m of each other, or provide an ipwidgets to allow users to choose the maximum walking distance interactively (up to 500 m)

# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def display_stops(stops: pd.DataFrame, edges: pd.DataFrame, distance):
    """
    stops: the table computed in a)
    edges: the table computed in b)
    """
    edge_df = edges[edges.distance<distance][edges.a_stop_id<edges.b_stop_id].reset_index()
    fig = go.Figure(px.scatter_map(
    stops,
    lat = 'stop_lat',
    lon = 'stop_lon',
    hover_name = 'stop_name',
    labels = {
        'stop_lat': 'Latitude',
        'stop_lon': 'Longitude',
    },
    title = "Stops and distances",
    opacity = 0.8,
    zoom = 11,
    width = 1000,
    height = 800,
    color_discrete_sequence = ["#ff0000"]
    ))
    
    # We build the lists here with 'None' to break the lines
    lons = []
    lats = []
    texts = []
    for i in range(len(edge_df)):
        lons.extend([edge_df['a_stop_lon'][i], edge_df['b_stop_lon'][i], None])
        lats.extend([edge_df['a_stop_lat'][i], edge_df['b_stop_lat'][i], None])
        texts.extend([edge_df['a_stop_name'][i], edge_df['b_stop_name'][i], None])
        
    # Your exact add_trace, but passed the lists instead of looping
    fig.add_trace(go.Scattermap(
        lon = lons,
        lat = lats,
        mode = "lines+markers",
        marker_color = "green",
        showlegend = False,
        hoverinfo = "text+lat+lon",
        hoverlabel = {"bgcolor": "red"},
        hovertext = texts
        )
                 )
    fig.show()


# %%
# Missing imports
import ipywidgets as widgets
from ipywidgets import fixed, interact
import pandas as pd

# Missing data
stops_df = pd.read_sql(f"SELECT * FROM {userns}.stops", conn)
edges_df = pd.read_sql(f"SELECT * FROM {userns}.stop_to_stop", conn)

m = interact(display_stops,
         distance=widgets.IntSlider(min=0, max=500, step=10, value=250),
         stops=fixed(stops_df),
         edges=fixed(edges_df)
)


# %% [markdown]
#  ### d) Create a table of _stop times_ in selected region - 5/50
#
#  * Find the stop times and weekdays of trips (trip_id) servicing the stops found previously (i.e. they service at least one of the stops).
#  * Use the stop times and calendar information published on the same week as the stops information used to compute the stops in the Lausanne region.
#  * Save the results in the table _{userns}.stop_times_
#
#  At a minimum, the table should be as follow. Use the provided information to decide the best types for the fields.
#
#  * _{userns}.stop_times_ (subset of _{sharedns}.sbb_stop_times_ and _{sharedns}.sbb_calendar_).
#      * _trip_id_
#      * _stop_id_
#      * _departure_time_
#      * _arrival_time_
#      * _monday_ (trip happens on Monday)
#      * _tuesday_
#      * _wednesday_
#      * _thursday_
#      * _friday_
#      * _saturday_
#      * _sunday_
#
#
#  ---
#
#  * Check your result: For instance look out for redundant _trip_id,stop_id pairs_ which may indicate the presence of a loop on the circuit.
#
#  * Pay special attention to the value ranges of the _departure_time_ and _arrival_time_ fields in the _{sharedns}.sbb_stop_times_ table.
#
#  * This new table will be used in the next exercise for a routing algorithm. We recommend reviewing it to determine the appropriate data types and potential transformations for the _departure_time_ and _arrival_time_ fields.
#
#
#
#
#
#
#
#
#

# %%
drop_stop_times = f"""
DROP TABLE IF EXISTS {userns}.stop_times"""
with closing(conn.cursor()) as cur:
    cur.execute(drop_stop_times)
create_stop_times = f""" CREATE TABLE {userns}.stop_times 
AS SELECT trip_id, arrival_time, departure_time, s.stop_id, pub_date, stop_sequence
FROM (SELECT 
trip_id,
arrival_time,
departure_time,
CAST(split_part(stop_id, ':', 1) AS INTEGER) AS stop_id,
pub_date, 
stop_sequence
FROM {sharedns}.sbb_stop_times WHERE pub_date = DATE '{max_pub_date}') s
JOIN
(SELECT stop_id FROM {userns}.stops) t
ON s.stop_id = t.stop_id
"""
with closing(conn.cursor()) as cur:
    cur.execute(create_stop_times)

# %%
drop_stop_times_trips = f"""
DROP TABLE IF EXISTS {userns}.stop_times_trips"""
with closing(conn.cursor()) as cur:
    cur.execute(drop_stop_times_trips)
create_stop_times_trips = f"""CREATE TABLE {userns}.stop_times_trips AS 
SELECT a.trip_id, arrival_time, departure_time, stop_id, pub_date, service_id, stop_sequence
FROM {userns}.stop_times a
JOIN
(SELECT trip_id, service_id FROM {sharedns}.sbb_trips WHERE pub_date = DATE '{max_pub_date}') b
ON a.trip_id = b.trip_id
"""
with closing(conn.cursor()) as cur:
    cur.execute(create_stop_times_trips)

# %%
drop_full = f"""
DROP TABLE IF EXISTS {userns}.stop_times"""
with closing(conn.cursor()) as cur:
    cur.execute(drop_full)
    
create_full = f"""CREATE TABLE {userns}.stop_times
AS SELECT *
FROM (SELECT 
    a.trip_id,
    arrival_time,
    departure_time,
    stop_id,
    a.pub_date,
    a.service_id,
    CAST(stop_sequence AS INT) AS stop_sequence,
    monday,
    tuesday,
    wednesday,
    thursday,
    friday,
    saturday,
    sunday
FROM {userns}.stop_times_trips a
JOIN
(SELECT *
FROM {sharedns}.sbb_calendar
WHERE pub_date = DATE '{max_pub_date}') b
on a.service_id = b.service_id) a
"""
with closing(conn.cursor()) as cur:
    cur.execute(create_full)

# %%
pd.read_sql(f"""SELECT MIN(departure_time) AS min_departure, MAX(departure_time) AS max_departure, MIN(arrival_time) AS min_arrival, MAX(arrival_time) AS max_arrival
FROM {userns}.stop_times
""", conn)


# %% [markdown]
# The maximum departure and arrival times are between 24 and 48 hours, indicating that currently, trips are stored so that both `departure_time` and `arrival_time` increase regardless of day change. While this may be convenient when considering single trips, the algorithm would have to treat time differently depending on the starting time of the trip. We will therefore convert every trip to the standard 24 hour format and change the days of the trips accordingly when necessary.

# %%
with closing(conn.cursor()) as cur:
    cur.execute(f"""ALTER TABLE {userns}.full_table
    ADD COLUMN IF NOT EXISTS temp BOOLEAN""")
    
with closing(conn.cursor()) as cur:
    cur.execute(f"""
    UPDATE {userns}.full_table
    SET
        temp = sunday,
        sunday = CASE
                    WHEN (sunday = TRUE and arrival_time < '24:00:00') OR (saturday = TRUE and arrival_time >= '24:00:00') THEN TRUE
                    ELSE FALSE
                    END,
        saturday = CASE
                        WHEN (saturday = TRUE and arrival_time < '24:00:00') OR (friday = TRUE and arrival_time >= '24:00:00') THEN TRUE
                        ELSE FALSE
                    END,
        friday =    CASE
                        WHEN (friday = TRUE and arrival_time < '24:00:00') OR (thursday = TRUE and arrival_time >= '24:00:00') THEN TRUE
                        ELSE FALSE
                    END,
        thursday =  CASE
                        WHEN (thursday = TRUE and arrival_time < '24:00:00') OR (wednesday = TRUE and arrival_time >= '24:00:00') THEN TRUE
                        ELSE FALSE
                    END,
        wednesday = CASE
                        WHEN (wednesday = TRUE and arrival_time < '24:00:00') OR (tuesday = TRUE and arrival_time >= '24:00:00') THEN TRUE
                        ELSE FALSE
                    END,
        tuesday =   CASE
                        WHEN (tuesday = TRUE and arrival_time < '24:00:00') OR (monday = TRUE and arrival_time >= '24:00:00') THEN TRUE
                        ELSE FALSE
                    END,      
        monday = CASE
                        WHEN (monday = TRUE and arrival_time < '24:00:00') OR (temp = TRUE and arrival_time >= '24:00:00') THEN TRUE 
                        ELSE FALSE
                    END,
        arrival_time = CONCAT(LPAD(CAST(MOD(CAST(SUBSTRING(arrival_time, 1, 2) AS INT), 24) AS VARCHAR), 2, '0'), SUBSTRING(arrival_time, 3)),
        departure_time = CONCAT(LPAD(CAST(MOD(CAST(SUBSTRING(departure_time, 1, 2) AS INT), 24) AS VARCHAR), 2, '0'), SUBSTRING(departure_time, 3))
    """)
with closing(conn.cursor()) as cur:
    cur.execute(f"""ALTER TABLE {userns}.full_table
    DROP COLUMN temp""")

# %% [markdown]
# ### e) Public Transport Routing - 20/50

# %% [markdown]
# Note: The next questions are open-ended, and credits will be allocated based on the quality of both the proposed algorithm and its implementation. You will receive credits for proposing a robust algorithm, even if you do not carry out the implementation.
#
# We now ask you to implement a routing algorithm that computes the path between two given stops in the region of interest, or from a given stop to all the other stops in the region.
#
#
# We categorize the actions a traveler may undertake while using public transport into three types:
#
# - Boarding a public transport vehicle from a stop (embarking on a trip).
# - Alighting from a public transport vehicle at a stop.
# - Walking between two stops.
#
# For evaluation purposes, you will create a function in a separate python program that can be imported and invoked as follows. You may use but don't have to use it in this notebook.
#
# ```python
# from com490 import JourneyPlanner
#
# # Create a new journey planner.
# # schema is the Trino Schema under which tables, views and materialize views
# # will be stored.
# jp = JourneyPlanner(schema=userns)
#
# # Recreate the public transport tables, etc required by the route method,
# # limited to stops in the given region(s), and only trips servicing these stops.
# # uuids are from the com490_iceberg.geo table. This will be called only once.
# jp.prepare(regions=[uuid_lausanne,uuid_ouest_lausannois])
#
# # Compute the routes, from the start id (numerical) to end_id or all stops in region
# # if end_id is None, departing on Monday at 12:30 and max walking distance between
# # any two stops 100m (it will never be more than 500m).
# route = jp.route(start_id=85012345, end_id=None, departs="12:30", day="monday", max_walk_m=100)
# ```
#
# The _route_ is represented as a chronologically ordered list of tuples, where each tuple records the earliest time _ts_ that a person can either reach or use a stop or trip when starting from _stad_id_ at _departs_ time:
#
# * _(ts, stop, trip_id, None)_: the person can board _trip_id_ at the given _stop_ at time ts (trip departure time).
#
# * _(ts, None, trip_id, stop)_: the person can arrive at _stop_ at time _ts_, using the given _trip_id_ (trip arrival time).
#
# * _(ts, stop_prev, None, stop)_: the person can reach stop via _stop_prev_ at time _ts_ if this provides a faster path.
#
# Note: Only the shortest paths are considered. Therefore, for a given departure from _start_id_ at time _departs_, there is exactly one earliest time that the person can reach a stop or board a trip.

# %% [markdown]
# <hr>

# %% [markdown]
# ### CSA Routing Algorithm
# For the routing component, we selected the **Connection Scan Algorithm (CSA)**.  
# Paper reference: [Connection Scan Algorithm (Dibbelt et al., 2017)](https://arxiv.org/pdf/1703.05997)
# CSA is well suited for timetable-based public transport routing because it works directly on a list of connections sorted by departure time and computes earliest-arrival journeys efficiently. This fits both the structure of the SBB timetable data and the assignment objective.
# The current implementation focuses on the earliest-arrival variant of CSA. Connections are scanned in increasing departure-time order. A connection is usable if its departure stop has already been reached before its departure time. When this happens, the arrival time of the destination stop is updated, and feasible walking transfers are propagated.
# Practical implementation choices include:
# - preprocessing the timetable into a dedicated connections table
# - converting times into integer seconds for faster comparisons
# - grouping connections by weekday
# - loading connections and footpaths in memory for fast query execution
# #### CSADataHandler module
# The CSADataHandler class prepares the timetable data for CSA and transforms the raw SBB tables into a routing-ready format.
# Its main responsibilities are:
# - retrieve the latest timetable publication date
# - filter stop times to the stops in the selected region
# - join stop times with trip and calendar information
# - preserve trip order through stop_sequence
# - create the connections table by linking each stop to its next stop in the same trip
# - convert departure and arrival times into integer second values
# - validate the final data by checking for broken sequences, negative travel times, null values, and duplicate edges
# At the current stage, the prepared data is loaded fully into memory, tested on regions consisting of Lausanne and Ouest lausannois.
#
# **Filtering anomalies**: as identified during the timetable exploration and validation step, a small number of consecutive-stop connections showed an apparent extra +24h offset, producing unrealistic travel times. We removed them at connection-building time by excluding only adjacent edges whose raw duration is at least 24 hours, but becomes plausible after subtracting exactly 24 hours, while preserving normal GTFS times beyond 24:00.
#
# #### JourneyPlanner
# The JourneyPlanner class executes the routing query once the data has been prepared.
# Its role is to:
# - load stops, footpaths, and precomputed connections
# - organize connections by weekday
# - initialize earliest-arrival labels
# - apply initial walking transfers from the source stop
# - scan the connection list in increasing departure-time order per day
# - update reachable stops and propagate walking transfers
# - reconstruct the route through predecessor tracking when a destination is provided
# - check for routes departing the following day if the destination has not been reached within the day
#   
# This keeps the implementation modular: CSADataHandler handles preprocessing and validation, while JourneyPlanner handles query-time routing.
# #### Future extensions
# The current implementation provides a clean CSA baseline that we intend to extended further. We plan to:
# - retrieving only the relevant chunks of connections for larger regions instead of loading the full data into memory
# - further optimizing the routing core using lower-level or compiled implementations such as C, if performance becomes a bottleneck
# - adding the CSA starting criterion and stopping criterion to avoid unnecessary scans
#

# %%
import importlib
from journey_planner import JourneyPlanner

schema = userns

jp = JourneyPlanner(schema=schema)

# prepare once
jp.prepare(
    regions=[
        "a7a21b73-6ffe-4fbf-a635-6e2b961f3072",  # Lausanne (district)
        "e168fd57-f57a-4075-a350-0dcfbb55147f",  # Ouest lausannois
    ],
    rebuild=True
)

# test route to ALL stops 
res_all = jp.route(
    start_id=8501120,
    end_id=None,
    departs="12:30",
    day="monday",
    max_walk_m=100
)

print(len(res_all))

# %%
# specific route
route = jp.route(
    start_id=8501120,
    end_id=8501117,
    departs="23:59",
    day="monday",
    max_walk_m=500
)

route

# %% [markdown]
# ### f) Isochrone Map - 10/50

# %% [markdown]
# **Question**:
# * Given a time of day on Monday (or any other day of the week you may choose), and a starting point.
# * Apply a routing algorithm to estimate the shortest time required to reach each stop within the region using public transport.
# * Visualize the outcomes through a heatmap (e.g., utilizing Plotly), where the color of each stop varies based on the estimated travel time from the specified starting point. See example:
#
# ![example](./figs/isochrone.png).
#
# - Focus first on scenarios where walking between stops is not permitted. Once an algorithm is established, walking can optionally be incorporated, assuming a walking speed of 50 meters per minute. Walking being optional, bonus points (+2) will be awarded for implementing it. 
# - If walking is not considered, a journey consists of a sequence of _stop_ids_, each separated by a corresponding trip_id, in chronological order. For example: stop-1, trip-1, stop-2, trip-2, ..., stop-n.
# - Connections between consecutive stops and trips can only occur at predetermined times. Each trip-id, stop-id pair must be unique and occur at a specific time on any given day according to the timetable. If you want to catch an earlier connection, you must have taken an earlier trip; you cannot go back in time once you've arrived at a stop.
# - When making your design decision, consider using a _label-correcting_ algorithm rather than a _label-setting_ algorithm.

# %%
from isochrone_map import load_stops_metadata, build_isochrone_dataframe

stops_df = stops_df = load_stops_metadata(jp.schema, conn)

def plot_isochrone(start_stop_name, end_stop_name, day, start_time_hour, start_time_min, max_walk, max_hours):
    # jp = jp
    # conn = conn
    departs = str(start_time_hour) + ":" + str(start_time_min)
    
    iso_df = build_isochrone_dataframe(
        jp=jp,
        conn=conn,
        start_stop_name=start_stop_name,
        end_stop_name=end_stop_name,
        departs=departs,
        day=day,
        max_walk_m=max_walk,
        max_hours=max_hours,
        stops_df=stops_df
    )

    fig = go.Figure(px.scatter_map(
    iso_df.iloc[1:],
    lat = "stop_lat",
    lon = "stop_lon",
    hover_name = "stop_name",
    labels = {
        "stop_lat": "Latitude",
        "stop_lon": "Longitude",
        "travel_min": "TravelTime"
    },
    color="travel_min",
    color_continuous_scale="Plasma",
    # range_color=(0, max_hours * 60),
    range_color=(0, 80),
    title = "Isochrone map",
    opacity = 0.8,
    zoom = 11,
    width = 1000,
    height = 800
    ))

    start = iso_df[iso_df["stop_name"] == start_stop_name].iloc[0]
    
    fig.add_trace(go.Scattermap(
        lat = [start["stop_lat"]],
        lon = [start["stop_lon"]],
        mode = "markers",
        marker = dict(
            size = 10,
            color = "red",
            symbol = "circle"
        ),
        text = [f"Start: {start['stop_name']}"],
        hovertemplate = (
            "<b>%{text}</b><br>" +
            "Latitude: %{lat:.5f}<br>" +
            "Longitude: %{lon:.5f}<extra></extra>"
        ),
        showlegend = False
    ))

    fig.show()


# %%
m = interact(plot_isochrone,
            start_stop_name=widgets.Dropdown(
                options=stops_df["stop_name"].sort_values(),
                value="Lausanne",
                description="Start"),
             end_stop_name=widgets.Dropdown(
                options=pd.concat([pd.Series([None]),
                                   stops_df["stop_name"].sort_values()
                                  ]),
                value=None,
                description="End"),
             day=widgets.Dropdown(options=["monday","tuesday","wednesday","thursday", "friday", "saturday", "sunday"],
                                  value="monday",
                                  description="Day"),
             start_time_hour=widgets.IntSlider(min=0, max=23, step=1, value=12, description="Start Hour"),
             start_time_min=widgets.IntSlider(min=0, max=59, step=1, value=30, description="Start Minute"),
             max_walk=widgets.IntSlider(min=0, max=500, step=1, value=250, description="Max Walk (m)"),
             max_hours=widgets.IntSlider(min=0, max=10, step=1, value=2, description="Max Hours")
            )

# %%

# %%

# %%

# %%
