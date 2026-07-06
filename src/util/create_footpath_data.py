#!/usr/bin/env python3
"""
One-time script: compute real walking distances between ALL Swiss transit stops
using the OSM pedestrian network and persist them as a Trino table.

The resulting table covers all of Switzerland. build_footpaths() in
CSADataHandler filters it to the current region at table-build time.

Usage
-----
    python src/util/create_footpath_data.py
    python src/util/create_footpath_data.py --max-walk 500 --batch-size 2000
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from contextlib import closing
from pathlib import Path


import numpy as np
import pandas as pd

_REPO_ROOT  = Path(__file__).parents[2]
_CROW_SLACK = 1.5  # walk paths are longer than crow-fly; give Dijkstra slack

# planet.osm.ch is hosted in Switzerland — much faster from EPFL than Geofabrik.
# Geofabrik is the fallback.
_OSM_URLS = [
    "https://planet.osm.ch/switzerland-padded.osm.pbf",
    "https://download.geofabrik.de/europe/switzerland-latest.osm.pbf",
]


# Switzerland PBF is ~400-550 MB; anything smaller is a partial/corrupt download.
_MIN_PBF_MB = 300


# Step 1 – Download OSM data
def download_osm(osm_dir: str) -> str:
    import shutil
    os.makedirs(osm_dir, exist_ok=True)
    pbf = os.path.join(osm_dir, "switzerland-latest.osm.pbf")

    if os.path.exists(pbf):
        size_mb = os.path.getsize(pbf) / 1_048_576
        if size_mb >= _MIN_PBF_MB:
            print(f"[1/5] Cached OSM file ({size_mb:.0f} MB): {pbf}")
            return pbf
        print(f"[1/5] Cached file is only {size_mb:.0f} MB (incomplete) — deleting and re-downloading…")
        os.remove(pbf)

    print(f"[1/5] Downloading Swiss OSM data (~400 MB, one-time)…")

    for i, url in enumerate(_OSM_URLS):
        print(f"      {url}")
        tmp = pbf + ".tmp"
        try:
            if shutil.which("wget"):
                subprocess.run(
                    ["wget", "--show-progress", "-O", tmp, url], check=True
                )
            elif shutil.which("curl"):
                subprocess.run(
                    ["curl", "-L", "--progress-bar", "-o", tmp, url], check=True
                )
            else:
                def _hook(count, block, total):
                    pct = min(100, int(count * block * 100 / total))
                    print(f"\r  {pct:3d}%  {count * block / 1_048_576:6.0f} MB", end="", flush=True)
                urllib.request.urlretrieve(url, tmp, _hook)
                print()
            os.replace(tmp, pbf)  # atomic rename — only appears if download completed
            break
        except Exception as exc:
            if os.path.exists(tmp):
                os.remove(tmp)
            if i < len(_OSM_URLS) - 1:
                print(f"      failed ({exc}), trying fallback URL…")
            else:
                raise

    return pbf


# Step 2 – Build walking graph from PBF using osmium (no gcc required)
def build_walk_graph(pbf_path: str):
    import networkx as nx
    import osmium

    _WALKABLE = frozenset({
        "footway", "pedestrian", "path", "steps", "crossing",
        "living_street", "residential", "service", "track",
        "cycleway", "unclassified", "tertiary", "tertiary_link",
        "secondary", "secondary_link", "primary", "primary_link", "road",
    })
    _MOTOR_ONLY = frozenset({"motorway", "motorway_link", "trunk", "trunk_link"})

    class _WalkHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.node_coords: dict[int, tuple[float, float]] = {}
            self.edge_triples: list[tuple[int, int, float]] = []

        def way(self, w):
            hw = w.tags.get("highway", "")
            if not hw:
                return
            foot = w.tags.get("foot", "")
            if hw in _MOTOR_ONLY:
                if foot != "yes":
                    return
            elif hw not in _WALKABLE:
                return
            if foot == "no":
                return
            valid = [n for n in w.nodes if n.location.valid()]
            for i in range(len(valid) - 1):
                a, b = valid[i], valid[i + 1]
                lat1, lon1 = a.location.lat, a.location.lon
                lat2, lon2 = b.location.lat, b.location.lon
                self.node_coords[a.ref] = (lat1, lon1)
                self.node_coords[b.ref] = (lat2, lon2)
                self.edge_triples.append((a.ref, b.ref, _haversine_m(lat1, lon1, lat2, lon2)))

    print("[2/5] Parsing OSM pedestrian network (several minutes, ~3–6 GB RAM)…")
    t0 = time.perf_counter()
    handler = _WalkHandler()
    handler.apply_file(pbf_path, locations=True)

    G = nx.Graph()
    for osmid, (lat, lon) in handler.node_coords.items():
        G.add_node(osmid, lat=lat, lon=lon)
    for u, v, length in handler.edge_triples:
        G.add_edge(u, v, length=length)

    print(
        f"      {G.number_of_nodes():,} nodes  "
        f"{G.number_of_edges():,} edges  "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    return G


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    import numpy as _np
    R   = 6_371_000.0
    phi1, phi2 = _np.radians(lat1), _np.radians(lat2)
    a = (
        _np.sin(_np.radians(lat2 - lat1) / 2) ** 2
        + _np.cos(phi1) * _np.cos(phi2) * _np.sin(_np.radians(lon2 - lon1) / 2) ** 2
    )
    return 2.0 * R * _np.arcsin(_np.sqrt(_np.clip(a, 0, 1)))


# Step 3 – Load all Swiss stops from shared schema
def load_all_stops(conn, shared_schema: str, max_pub_date: str) -> pd.DataFrame:
    print(f"[3/5] Loading all Swiss stops (pub_date={max_pub_date})…")
    query = f"""
        SELECT stop_id, stop_lat, stop_lon FROM (
            SELECT
                TRY_CAST(split_part(stop_id, ':', 1) AS INTEGER) AS stop_id,
                AVG(stop_lat) AS stop_lat,
                AVG(stop_lon) AS stop_lon
            FROM {shared_schema}.sbb_stops
            WHERE pub_date = DATE '{max_pub_date}'
              AND split_part(stop_id, ':', 1) LIKE '85%'
              AND stop_lat IS NOT NULL
              AND stop_lon IS NOT NULL
            GROUP BY TRY_CAST(split_part(stop_id, ':', 1) AS INTEGER)
        )
        WHERE stop_id IS NOT NULL
    """
    df = pd.read_sql(query, conn).drop_duplicates("stop_id").dropna().reset_index(drop=True)
    print(f"      {len(df):,} unique stops")
    return df


def _get_max_pub_date(conn, shared_schema: str) -> str:
    df = pd.read_sql(
        f"SELECT MAX(pub_date) AS d FROM {shared_schema}.sbb_stop_times", conn
    )
    return str(df.iloc[0]["d"])[:10]


# Step 4 – Snap stops to OSM nodes via KD-tree
def snap_stops(G, stops_df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    from scipy.spatial import cKDTree

    print("[4/5] Snapping stops to OSM nodes…")
    node_ids   = np.array(list(G.nodes()))
    node_attrs = [G.nodes[n] for n in node_ids]
    lats = np.array([d.get("lat", d.get("y", np.nan)) for d in node_attrs])
    lons = np.array([d.get("lon", d.get("x", np.nan)) for d in node_attrs])
    valid    = ~(np.isnan(lats) | np.isnan(lons))
    node_ids = node_ids[valid]
    lats     = lats[valid]
    lons     = lons[valid]
    tree = cKDTree(np.column_stack([lats, lons]))
    _, idx = tree.query(stops_df[["stop_lat", "stop_lon"]].values)
    result = stops_df.copy()
    result["osm_node"] = node_ids[idx]
    return result


# Step 5 – Compute walking distances via single-source Dijkstra with cutoff
def compute_footpaths(
    G,
    stops_df: pd.DataFrame,
    max_walk_m: float,
    walk_speed: float,
    walk_base: float,
) -> pd.DataFrame:
    import numpy as np
    import networkx as nx
    from scipy.spatial import cKDTree
    from tqdm import tqdm

    stops     = stops_df.reset_index(drop=True)
    lats      = stops["stop_lat"].values
    lons      = stops["stop_lon"].values
    osm_nodes = stops["osm_node"].values
    stop_ids  = stops["stop_id"].values
    cutoff_m  = max_walk_m * _CROW_SLACK
    deg_r     = cutoff_m / 111_320.0
    coord_kd  = cKDTree(np.column_stack([lats, lons]))

    print(f"[5/5] Walking distances for {len(stops):,} stops (cutoff={cutoff_m:.0f} m)…")
    records: list[tuple] = []
    for i in tqdm(range(len(stops))):
        candidates = coord_kd.query_ball_point([lats[i], lons[i]], deg_r)
        try:
            reach = nx.single_source_dijkstra_path_length(
                G, osm_nodes[i], cutoff=cutoff_m, weight="length"
            )
        except nx.NodeNotFound:
            continue
        for j in candidates:
            if j == i or osm_nodes[j] not in reach:
                continue
            d = reach[osm_nodes[j]]
            if d <= max_walk_m:
                records.append((
                    int(stop_ids[i]),
                    int(stop_ids[j]),
                    float(d),
                    float(walk_base + d / walk_speed),
                ))

    return pd.DataFrame(records, columns=["a_stop_id", "b_stop_id", "distance", "walk_time_min"])


# Step 6 – Upload to Trino via batched INSERT
def upload_to_trino(conn, table: str, df: pd.DataFrame, batch_size: int) -> None:
    from tqdm import tqdm
    print(f"Uploading {len(df):,} rows → {table}  (batch_size={batch_size})…")
    with closing(conn.cursor()) as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute(f"""
            CREATE TABLE {table} (
                a_stop_id     INTEGER,
                b_stop_id     INTEGER,
                distance      DOUBLE,
                walk_time_min DOUBLE
            )
        """)

    batches = [df.iloc[i : i + batch_size] for i in range(0, len(df), batch_size)]
    with closing(conn.cursor()) as cur:
        for batch in tqdm(batches, desc="  inserting"):
            vals = ", ".join(
                f"({int(r.a_stop_id)}, {int(r.b_stop_id)}, {r.distance:.4f}, {r.walk_time_min:.6f})"
                for r in batch.itertuples(index=False)
            )
            cur.execute(
                f"INSERT INTO {table} (a_stop_id, b_stop_id, distance, walk_time_min) VALUES {vals}"
            )
    print(f"Done – {table}")


def _ensure_deps() -> None:
    # osmium (pyosmium) ships pre-compiled manylinux wheels — no gcc required.
    # pyrosm was replaced because its dependencies (cykhash, pyrobuf) need gcc.
    needed = [
        ("osmium",   "osmium"),
        ("scipy",    "scipy"),
        ("tqdm",     "tqdm"),
        ("networkx", "networkx"),
    ]
    for mod, pkg in needed:
        try:
            __import__(mod)
        except ImportError:
            print(f"  pip install {pkg}…")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
            except subprocess.CalledProcessError:
                raise ImportError(
                    f"Auto-install of '{pkg}' failed. Install manually: pip install {pkg}"
                )


def main(
    max_walk_m: float = 500.0,
    walk_speed: float = 50.0,
    walk_base:  float = 2.0,
    batch_size: int   = 2000,
) -> None:
    _ensure_deps()

    sys.path.insert(0, str(_REPO_ROOT))
    from src.config.settings import get_settings
    from src.config.trino_connection import create_trino_connection

    settings = get_settings()
    conn     = create_trino_connection(settings)

    osm_dir         = str(_REPO_ROOT / "data" / "osm")
    footpaths_table = f"{settings.user_schema}.footpaths_osm"
    max_pub_date    = _get_max_pub_date(conn, settings.shared_schema)

    pbf      = download_osm(osm_dir)
    G        = build_walk_graph(pbf)
    stops_df = load_all_stops(conn, settings.shared_schema, max_pub_date)
    stops_df = snap_stops(G, stops_df)
    fp_df    = compute_footpaths(G, stops_df, max_walk_m, walk_speed, walk_base)

    upload_to_trino(conn, footpaths_table, fp_df, batch_size)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-walk",    type=float, default=500.0, help="Max walking distance in metres")
    p.add_argument("--walk-speed",  type=float, default=50.0,  help="Walking speed in m/min")
    p.add_argument("--walk-base",   type=float, default=2.0,   help="Base transfer time in minutes")
    p.add_argument("--batch-size",  type=int,   default=2000,  help="Rows per Trino INSERT")
    args = p.parse_args()
    main(args.max_walk, args.walk_speed, args.walk_base, args.batch_size)
