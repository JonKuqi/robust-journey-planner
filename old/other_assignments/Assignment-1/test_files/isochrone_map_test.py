from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# import folium
import pandas as pd
from branca.colormap import linear
import ipywidgets as widgets
from IPython.display import display


@dataclass
class IsochroneConfig:
    day: str = "monday"
    departs: str = "12:30"
    max_walk_m: int = 100
    max_hours: int = 2
    zoom_start: int = 12


def load_stops_metadata(schema: str, conn) -> pd.DataFrame:
    """
    Load stop metadata needed for map rendering.

    Assumes the prepared stops table contains:
      - stop_id
      - stop_name
      - stop_lat
      - stop_lon
    """
    query = f"""
    SELECT
        stop_id,
        stop_name,
        stop_lat,
        stop_lon
    FROM {schema}.stops
    """
    return pd.read_sql(query, conn)


def build_isochrone_dataframe(
    jp,
    conn,
    start_id: int,
    end_id: int,
    departs: str = "12:30",
    day: str = "monday",
    max_walk_m: int = 100,
) -> pd.DataFrame:
    """
    Run one-to-all routing and return a dataframe with travel times + stop metadata.
    """
    arrivals = jp.route(
        start_id=start_id,
        end_id=end_id,
        departs=departs,
        day=day,
        max_walk_m=max_walk_m,
    )

    departure_secs = jp._time_to_secs(departs)

    arrivals_df = pd.DataFrame(
        [
            {
                "stop_id": int(stop_id),
                "arrival_secs": int(arr_secs),
                "travel_min": (int(arr_secs) - departure_secs) / 60.0,
            }
            for stop_id, arr_secs in arrivals.items()
        ]
    )

    stops_df = load_stops_metadata(jp.schema, conn)

    iso_df = arrivals_df.merge(stops_df, on="stop_id", how="left")
    iso_df = iso_df.dropna(subset=["stop_lat", "stop_lon"]).copy()
    iso_df = iso_df.sort_values("travel_min").reset_index(drop=True)

    return iso_df


def run_isochrone_map(
    jp,
    conn,
    start_id: int,
    departs: str = "12:30",
    day: str = "monday",
    max_walk_m: int = 100,
    max_hours: int = 2,
    zoom_start: int = 12,
):
    """
    One-shot helper:
      1. run one-to-all routing
      2. build dataframe
      3. return folium map
    """
    iso_df = build_isochrone_dataframe(
        jp=jp,
        conn=conn,
        start_id=start_id,
        departs=departs,
        day=day,
        max_walk_m=max_walk_m,
    )

    return make_isochrone_map(
        iso_df=iso_df,
        start_id=start_id,
        max_hours=max_hours,
        zoom_start=zoom_start,
    )


def build_stop_dropdown_options(schema: str, conn):
    stops_df = load_stops_metadata(schema, conn)
    stops_df = stops_df.dropna(subset=["stop_name"]).sort_values("stop_name")

    return [
        (f"{row.stop_name} ({int(row.stop_id)})", int(row.stop_id))
        for row in stops_df.itertuples(index=False)
    ]


def display_isochrone_widget(jp, conn, default_start_id: Optional[int] = None):
    """
    Interactive notebook widget for exploring isochrones.
    """
    stop_options = build_stop_dropdown_options(jp.schema, conn)

    if default_start_id is None and stop_options:
        default_start_id = stop_options[0][1]

    start_stop_widget = widgets.Dropdown(
        options=stop_options,
        value=default_start_id,
        description="start_stop",
        layout=widgets.Layout(width="500px"),
    )

    hour_widget = widgets.IntSlider(value=12, min=0, max=23, step=1, description="hour")
    minute_widget = widgets.IntSlider(value=30, min=0, max=59, step=1, description="minute")
    walk_widget = widgets.IntSlider(value=100, min=0, max=500, step=25, description="max_walk")
    max_hours_widget = widgets.IntSlider(value=2, min=1, max=5, step=1, description="max_hours")

    run_button = widgets.Button(description="Run Interact")
    output = widgets.Output()

    def on_click(_):
        output.clear_output()
        with output:
            departs = f"{hour_widget.value:02d}:{minute_widget.value:02d}"
            m = run_isochrone_map(
                jp=jp,
                conn=conn,
                start_id=start_stop_widget.value,
                departs=departs,
                day="monday",
                max_walk_m=walk_widget.value,
                max_hours=max_hours_widget.value,
            )
            display(m)

    run_button.on_click(on_click)

    display(
        widgets.VBox(
            [
                start_stop_widget,
                hour_widget,
                minute_widget,
                walk_widget,
                max_hours_widget,
                run_button,
                output,
            ]
        )
    )
