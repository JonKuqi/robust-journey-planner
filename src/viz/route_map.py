from __future__ import annotations

"""Small Jupyter-friendly route visualization helpers."""


def route_to_dataframe(route: dict):
    """Convert a route dictionary into a pandas DataFrame of steps."""
    import pandas as pd

    rows = []
    for idx, step in enumerate(route.get("steps", [])):
        rows.append(
            {
                "step": idx,
                "type": step.get("type"),
                "line_text": step.get("line_text"),
                "from_stop": step.get("from_stop"),
                "to_stop": step.get("to_stop"),
                "departure_time": step.get("departure_time"),
                "arrival_time": step.get("arrival_time"),
                "predicted_delay_sec": step.get("predicted_delay_sec", 0),
                "robust_arrival_time": step.get("robust_arrival_time"),
            }
        )
    return pd.DataFrame(rows)


def plot_route_map(route: dict, stop_metadata: dict[int, dict] | None = None):
    """Render a simple folium map for a route when stop coordinates are available."""
    try:
        import folium
    except ImportError as exc:
        raise RuntimeError("Install folium to use plot_route_map().") from exc

    stop_metadata = stop_metadata or {}
    points = []
    for step in route.get("steps", []):
        for stop_key in ("from_stop", "to_stop"):
            stop_id = step.get(stop_key)
            meta = stop_metadata.get(stop_id, {})
            lat = meta.get("stop_lat")
            lon = meta.get("stop_lon")
            if lat is not None and lon is not None:
                points.append((float(lat), float(lon), stop_id))

    if not points:
        raise ValueError("No stop coordinates available for this route.")

    center = (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    fmap = folium.Map(location=center, zoom_start=13)
    folium.PolyLine([(lat, lon) for lat, lon, _ in points], weight=4, opacity=0.8).add_to(fmap)

    for lat, lon, stop_id in points:
        folium.CircleMarker(location=(lat, lon), radius=4, popup=str(stop_id), fill=True).add_to(fmap)

    return fmap

