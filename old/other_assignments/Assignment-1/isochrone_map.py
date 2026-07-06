import pandas as pd


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
    stops_df: pd.DataFrame,
    start_stop_name: str,
    end_stop_name: str,
    departs: str = "12:30",
    day: str = "monday",
    max_walk_m: int = 100,
    max_hours: int = 10
) -> pd.DataFrame:
    """
    Run one-to-all routing and return a dataframe with travel times + stop metadata.
    """

    start_id = stops_df[stops_df["stop_name"]==start_stop_name]["stop_id"].iloc[0]
    end_id = None if end_stop_name == None else stops_df[stops_df["stop_name"]==end_stop_name]["stop_id"].iloc[0]
    
    arrivals = jp.route(
        start_id=start_id,
        end_id=end_id,
        departs=departs,
        day=day,
        max_walk_m=max_walk_m,
    )

    departure_secs = jp._time_to_secs(departs)

    if type(arrivals) == dict:
        arrivals_df = pd.DataFrame(
                        [
                            {
                                "stop_id": int(stop_id),
                                "arrival_secs": int(arr_secs),
                                "travel_min": (int(arr_secs) - departure_secs) / 60.0,
                            }
                            for stop_id, arr_secs in arrivals.items()
                        ])
          
    else:
        arrivals_df = pd.concat([
            pd.DataFrame(
                            {
                                "stop_id": [start_id],
                                "arrival_secs": [0],
                                "travel_min": [0],
                            }),
            pd.DataFrame(
                        [
                            {
                                "stop_id": int(stop_id),
                                "arrival_secs": int(arr_secs),
                                "travel_min": (int(arr_secs) - departure_secs) / 60.0
                            }
                            for (arr_secs, start_id, trip_id, stop_id) in arrivals if stop_id != None
                        ])
        ])

    iso_df = arrivals_df.merge(stops_df, on="stop_id", how="left")
    iso_df = iso_df.dropna(subset=["stop_lat", "stop_lon"]).copy()
    iso_df = iso_df[iso_df["travel_min"] <= max_hours * 60]
    iso_df = iso_df.sort_values("travel_min").reset_index(drop=True)

    return iso_df
