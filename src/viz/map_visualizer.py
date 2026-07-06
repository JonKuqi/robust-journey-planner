"""Interactive map visualizer for journey planner output.

Self-contained widget: pre-fetches Swiss stop coordinates once from the shared
timetable, then offers a Jupyter UI with stop search, query controls, a
clickable route list, and an OpenStreetMap canvas drawing each leg in the
colour of its transport mode (walks dashed).

Stack: ipyleaflet + ipywidgets + pandas.
Usage:
    from src.viz.map_visualizer import MapVisualizer
    viz = MapVisualizer(planner=planner, settings=settings)
    viz.show()
"""
from __future__ import annotations

import datetime as _dt
import html as _html
from typing import Any

import pandas as pd


# GTFS-style transport codes → display colour
_MODE_COLORS = {
    "walk": "#666666",
    "B":    "#43a047",  # Bus
    "BUS":  "#43a047",
    "T":    "#fbc02d",  # Tram
    "TRA":  "#fbc02d",
    "M":    "#d32f2f",  # Metro
    "MET":  "#d32f2f",
    "S":    "#1976d2",  # S-Bahn
    "R":    "#1976d2",  # Regio
    "RE":   "#1565c0",  # RegioExpress
    "IR":   "#0d47a1",  # InterRegio
    "IC":   "#01579b",  # InterCity
    "ICE":  "#01579b",
    "EC":   "#01579b",
    "TER":  "#1976d2",
    "BAT":  "#0097a7",  # Boat
    "FUN":  "#6d4c41",  # Funicular
    "GB":   "#6d4c41",  # Gondola
}
_DEFAULT_MODE_COLOR = "#37474f"
_CH_CENTER = (46.8, 8.2)  # rough centre of Switzerland for the initial view


def _step_color(step: dict) -> str:
    if step.get("type") == "walk":
        return _MODE_COLORS["walk"]
    code = (step.get("transport") or "").upper()
    return _MODE_COLORS.get(code, _DEFAULT_MODE_COLOR)


def _fmt_walk(m: float) -> str:
    return f"{m:.0f} m" if m else "0 m"


class MapVisualizer:
    """Plan a route from the UI and visualise it on an interactive map."""

    def __init__(
        self,
        planner,
        conn=None,
        settings=None,
        stops_df: pd.DataFrame | None = None,
        planner_api: str = "robust",
        travel_date: str | _dt.date | None = None,
    ):
        """Args:
            planner:  A prepared planner instance.
            conn:     Trino DB-API connection. Used to fetch stops if `stops_df` is None.
            settings: `ProjectSettings`. Used together with `conn` for the stops fetch.
            stops_df: Optional pre-fetched DataFrame with columns
                      stop_id, stop_name, stop_lat, stop_lon.
            planner_api:
                "robust" / "robust_journey_planner" calls `planner.plan(...)`.
                "journey_planner_v2" calls `planner.plan_candidates(...)` directly.
            travel_date: Initial date shown in the date picker (string "YYYY-MM-DD" or
                         datetime.date). Defaults to today when omitted.
        """
        self.planner = planner
        self.planner_api = self._normalize_planner_api(planner_api)

        if isinstance(travel_date, str):
            travel_date = _dt.date.fromisoformat(travel_date)
        self._travel_date: _dt.date = travel_date or _dt.date.today()

        if stops_df is None:
            stops_df = self._load_swiss_stops(conn, settings)

        self._stops = (
            stops_df.dropna(subset=["stop_lat", "stop_lon"])
            .drop_duplicates(subset=["stop_id"])
            .set_index("stop_id")
        )
        self._search_df = (
            stops_df.dropna(subset=["stop_lat", "stop_lon"])
            .sort_values("stop_name")
            .reset_index(drop=True)
        )

        self._routes: list[dict] = []
        self._selected_idx: int = 0
        self._route_layers: list = []
        self._route_buttons: list = []

    @staticmethod
    def _normalize_planner_api(planner_api: str) -> str:
        api = str(planner_api or "robust").strip().lower()
        aliases = {
            "robust": "robust",
            "robust_journey_planner": "robust",
            "plan": "robust",
            "v2": "journey_planner_v2",
            "journey_planner_v2": "journey_planner_v2",
            "plan_candidates": "journey_planner_v2",
        }
        if api not in aliases:
            valid = ", ".join(sorted(set(aliases)))
            raise ValueError(f"Unknown planner_api '{planner_api}'. Expected one of: {valid}")
        return aliases[api]

    # ── data loading ──────────────────────────────────────────────────────
    @staticmethod
    def _load_swiss_stops(conn, settings) -> pd.DataFrame:
        if settings is None:
            from src.config.settings import get_settings
            settings = get_settings()
        if conn is None:
            from src.config.trino_connection import create_trino_connection
            conn = create_trino_connection(settings)

        max_pub = pd.read_sql(
            f"SELECT MAX(pub_date) AS d FROM {settings.shared_schema}.sbb_stop_times",
            conn,
        ).iloc[0]["d"]
        return pd.read_sql(
            f"""
            SELECT stop_id, stop_name, stop_lat, stop_lon FROM (
                SELECT
                    TRY_CAST(split_part(stop_id, ':', 1) AS INTEGER) AS stop_id,
                    MAX(stop_name) AS stop_name,
                    AVG(stop_lat)  AS stop_lat,
                    AVG(stop_lon)  AS stop_lon
                FROM {settings.shared_schema}.sbb_stops
                WHERE pub_date = DATE '{max_pub}'
                  AND split_part(stop_id, ':', 1) LIKE '85%'
                  AND stop_lat IS NOT NULL AND stop_lon IS NOT NULL
                GROUP BY TRY_CAST(split_part(stop_id, ':', 1) AS INTEGER)
            ) WHERE stop_id IS NOT NULL
            """,
            conn,
        )

    # ── lookups ───────────────────────────────────────────────────────────
    def _coord(self, stop_id) -> tuple[float, float] | None:
        try:
            sid = int(stop_id)
        except (TypeError, ValueError):
            return None
        if sid not in self._stops.index:
            return None
        row = self._stops.loc[sid]
        return (float(row["stop_lat"]), float(row["stop_lon"]))

    def _name(self, stop_id) -> str:
        try:
            sid = int(stop_id)
        except (TypeError, ValueError):
            return f"#{stop_id}"
        if sid not in self._stops.index:
            return f"#{sid}"
        return str(self._stops.loc[sid]["stop_name"])

    # ── public entry point ────────────────────────────────────────────────
    def show(self):
        """Build the interactive widget. Returns the root widget; display it
        as the last expression in a cell or via `IPython.display.display`."""
        try:
            import ipywidgets as W
            import ipyleaflet as L
        except ImportError as exc:
            raise RuntimeError(
                "Install ipyleaflet + ipywidgets:  pip install ipyleaflet ipywidgets"
            ) from exc

        self._from_search, self._from_select, from_box = self._build_stop_picker("From")
        self._to_search,   self._to_select,   to_box   = self._build_stop_picker("To")

        self._date_widget = W.DatePicker(
            description="Date:", value=self._travel_date,
            layout=W.Layout(width="220px"),
        )
        self._time_widget = W.Text(
            value="09:00", description="Arrive by:", placeholder="HH:MM",
            layout=W.Layout(width="200px"),
        )
        self._conf_widget = W.FloatSlider(
            value=0.9, min=0.5, max=0.95, step=0.05,
            description="Conf. Q:", readout_format=".2f",
            layout=W.Layout(width="280px"),
        )
        self._n_widget = W.IntSlider(
            value=5, min=1, max=10, step=1,
            description="Top N:", layout=W.Layout(width="240px"),
        )
        self._window_widget = W.IntSlider(
            value=180, min=30, max=600, step=30,
            description="Window:", readout_format="d",
            layout=W.Layout(width="280px"),
        )
        self._plan_button = W.Button(
            description="Plan route", icon="search", button_style="primary",
            layout=W.Layout(width="160px"),
        )
        self._plan_button.on_click(self._on_plan_clicked)
        self._status_html = W.HTML(value="")

        controls = W.VBox([
            W.HBox([from_box, to_box]),
            W.HBox([self._date_widget, self._time_widget, self._conf_widget, self._n_widget]),
            W.HBox([self._window_widget, self._plan_button, self._status_html]),
        ])

        self._map = L.Map(
            center=_CH_CENTER, zoom=8, scroll_wheel_zoom=True,
            layout=W.Layout(width="720px", height="540px"),
        )
        self._map.add_control(L.ScaleControl(position="bottomleft"))

        self._route_list_box = W.VBox(
            [],
            layout=W.Layout(
                width="320px", height="540px",
                overflow_y="auto", border="1px solid #ddd", padding="4px",
            ),
        )
        self._details_html = W.HTML(
            value=self._empty_details_html(),
            layout=W.Layout(width="100%", padding="8px",
                            border="1px solid #eee", margin="6px 0 0 0"),
        )

        main_row = W.HBox([self._route_list_box, self._map])
        return W.VBox([controls, main_row, self._details_html])

    # ── stop picker ───────────────────────────────────────────────────────
    def _build_stop_picker(self, label: str):
        import ipywidgets as W
        search = W.Text(
            placeholder=f"Type to search {label.lower()} stop…",
            description=f"{label}:",
            layout=W.Layout(width="380px"),
        )
        select = W.Select(
            options=[], rows=5, layout=W.Layout(width="380px"),
        )

        def on_change(change):
            q = (change["new"] or "").strip().lower()
            if len(q) < 2:
                select.options = []
                return
            mask = self._search_df["stop_name"].str.lower().str.contains(q, na=False, regex=False)
            matches = self._search_df.loc[mask].head(50)
            opts = [
                (f"{row.stop_name}  —  {int(row.stop_id)}", int(row.stop_id))
                for row in matches.itertuples(index=False)
            ]
            select.options = opts
            if opts:
                select.value = opts[0][1]

        search.observe(on_change, names="value")
        return search, select, W.VBox([search, select])

    # ── plan + status ─────────────────────────────────────────────────────
    def _on_plan_clicked(self, _btn):
        start_id = self._from_select.value
        end_id   = self._to_select.value
        if start_id is None or end_id is None:
            self._set_status("Pick both From and To stops first.", error=True)
            return
        if start_id == end_id:
            self._set_status("From and To are the same stop.", error=True)
            return

        date_val = self._date_widget.value
        if not isinstance(date_val, _dt.date):
            self._set_status("Pick a valid date.", error=True)
            return

        try:
            self._set_status("Planning…")
            self._plan_button.disabled = True
            routes = self._plan_routes(
                start_stop_id=int(start_id),
                end_stop_id=int(end_id),
                travel_date=str(date_val),
                arrival_deadline=self._time_widget.value,
                confidence_q=float(self._conf_widget.value),
                max_routes=int(self._n_widget.value),
                search_window_minutes=int(self._window_widget.value),
            )
        except Exception as e:
            self._set_status(f"Planner error: {_html.escape(str(e))}", error=True)
            self._plan_button.disabled = False
            return
        finally:
            self._plan_button.disabled = False

        self._routes = routes or []
        if not self._routes:
            self._set_status("No routes satisfy the constraints.", error=True)
            self._route_list_box.children = ()
            self._clear_route_layers()
            self._details_html.value = self._empty_details_html()
            return

        self._set_status(
            f"{len(self._routes)} route{'s' if len(self._routes) != 1 else ''} found."
        )
        self._rebuild_route_buttons()
        self._select_route(0)

    def _plan_routes(
        self,
        *,
        start_stop_id: int,
        end_stop_id: int,
        travel_date: str,
        arrival_deadline: str,
        confidence_q: float,
        max_routes: int,
        search_window_minutes: int = 180,
    ) -> list[dict]:
        return self.planner.plan(
            start_stop_id=start_stop_id,
            end_stop_id=end_stop_id,
            travel_date=travel_date,
            arrival_deadline=arrival_deadline,
            confidence_q=confidence_q,
            max_routes=max_routes,
            search_window_minutes=search_window_minutes,
        )

    def _set_status(self, msg: str, error: bool = False):
        color = "#c62828" if error else "#2e7d32"
        self._status_html.value = (
            f'<span style="color:{color};margin-left:12px;">{msg}</span>'
        )

    # ── route list ────────────────────────────────────────────────────────
    def _rebuild_route_buttons(self):
        import ipywidgets as W
        self._route_buttons = []
        children = []
        for i, route in enumerate(self._routes):
            btn = W.Button(
                description=self._route_summary_text(i, route),
                layout=W.Layout(width="300px", height="46px", margin="2px 0"),
                tooltip="Click to view this route on the map",
            )
            btn.on_click(lambda b, idx=i: self._select_route(idx))
            self._route_buttons.append(btn)
            children.append(btn)
        self._route_list_box.children = tuple(children)

    @staticmethod
    def _route_summary_text(i: int, route: dict) -> str:
        dep   = route.get("departure_time", "?")
        arr   = route.get("arrival_time", "?")
        dur   = int(route.get("duration_sec", 0) or 0) // 60
        nt    = int(route.get("n_transfers", 0) or 0)
        walk  = int(route.get("total_walk_m", 0) or 0)
        conf  = float(route.get("route_confidence", 0) or 0) * 100
        ok    = route.get("passes_confidence")
        badge = ("✓" if ok else "✗") if ok is not None else "·"
        return (
            f"#{i+1} {badge}  {dep}→{arr}  "
            f"{dur}min  {nt}x  {walk}m  {conf:.0f}%"
        )

    def _update_button_styles(self):
        for i, btn in enumerate(self._route_buttons):
            if i == self._selected_idx:
                btn.button_style = "primary"
                continue
            route = self._routes[i]
            passes = route.get("passes_confidence")
            conf   = float(route.get("route_confidence", 1.0) or 1.0)
            if passes is False or conf < 0.5:
                btn.button_style = "warning"
            else:
                btn.button_style = ""

    # ── selection + drawing ───────────────────────────────────────────────
    def _select_route(self, idx: int):
        if not (0 <= idx < len(self._routes)):
            return
        self._selected_idx = idx
        route = self._routes[idx]
        self._clear_route_layers()
        self._draw_route(route)
        self._update_button_styles()
        self._details_html.value = self._route_details_html(route)
        self._fit_to_route(route)

    def _clear_route_layers(self):
        for layer in self._route_layers:
            try:
                self._map.remove_layer(layer)
            except Exception:
                pass
        self._route_layers = []

    def _draw_route(self, route: dict):
        import ipyleaflet as L
        steps = route.get("steps") or []
        if not steps:
            steps = [{
                "type": "ride",
                "from_stop": route["start_stop"],
                "to_stop":   route["end_stop"],
                "transport": "",
                "line_text": "",
            }]

        for step in steps:
            a = self._coord(step.get("from_stop"))
            b = self._coord(step.get("to_stop"))
            if a is None or b is None or a == b:
                continue
            color = _step_color(step)
            dash  = "8,8" if step.get("type") == "walk" else None
            poly = L.Polyline(
                locations=[a, b],
                color=color,
                weight=6,
                opacity=0.85,
                dash_array=dash,
                fill=False,
            )
            self._map.add_layer(poly)
            self._route_layers.append(poly)

        # start / end markers
        start_id = route.get("start_stop")
        end_id   = route.get("end_stop")
        sc, ec = self._coord(start_id), self._coord(end_id)
        if sc is not None:
            self._add_marker(sc, self._name(start_id), color="green", icon="play")
        if ec is not None:
            self._add_marker(ec, self._name(end_id), color="red",   icon="flag-checkered")

        # transfer markers — at each step boundary except start/end
        for prev, nxt in zip(steps, steps[1:]):
            mid_id = nxt.get("from_stop")
            mc = self._coord(mid_id)
            if mc is None or mc == sc or mc == ec:
                continue
            tm = L.CircleMarker(
                location=mc, radius=7,
                color="#1a237e", weight=2,
                fill_color="#ffd54f", fill_opacity=1.0,
                title=self._name(mid_id),
            )
            self._map.add_layer(tm)
            self._route_layers.append(tm)

    def _add_marker(self, loc, name, *, color: str, icon: str):
        import ipyleaflet as L
        try:
            ic = L.AwesomeIcon(name=icon, marker_color=color, icon_color="white")
            m  = L.Marker(location=loc, icon=ic, draggable=False, title=name)
        except Exception:
            # Some ipyleaflet versions do not expose AwesomeIcon — fall back to circle
            m = L.CircleMarker(location=loc, radius=10, color=color, fill_color=color,
                               fill_opacity=1.0, title=name)
        self._map.add_layer(m)
        self._route_layers.append(m)

    def _fit_to_route(self, route: dict):
        coords: list[tuple[float, float]] = []
        for step in (route.get("steps") or []):
            for key in ("from_stop", "to_stop"):
                c = self._coord(step.get(key))
                if c is not None:
                    coords.append(c)
        for key in ("start_stop", "end_stop"):
            c = self._coord(route.get(key))
            if c is not None:
                coords.append(c)
        if not coords:
            return
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        sw = (min(lats) - 0.002, min(lons) - 0.002)
        ne = (max(lats) + 0.002, max(lons) + 0.002)
        if sw == ne:
            self._map.center = sw
            self._map.zoom   = 15
        else:
            self._map.fit_bounds([list(sw), list(ne)])

    # ── details panel ─────────────────────────────────────────────────────
    @staticmethod
    def _empty_details_html() -> str:
        return ('<div style="color:#888;padding:12px;text-align:center;">'
                'Pick a From and To stop, then click <b>Plan route</b>.'
                '</div>')

    def _route_details_html(self, route: dict) -> str:
        # Build per-step headway index from per_step_probabilities for delay display
        step_headways: dict[int, float] = {}
        for row in (route.get("per_step_probabilities") or []):
            idx = row.get("step_index")
            if idx is not None:
                step_headways[int(idx)] = float(row.get("headway_sec", 0) or 0)

        rows = []
        steps = route.get("steps") or []
        for si, step in enumerate(steps):
            from_name = self._name(step.get("from_stop"))
            to_name   = self._name(step.get("to_stop"))
            mode_type = step.get("type", "ride")
            color     = _step_color(step)
            dep       = step.get("departure_time", "")
            arr       = step.get("arrival_time", "")
            rob_arr   = step.get("robust_arrival_time", "")
            dur_min   = int(step.get("duration_sec", 0) or 0) // 60
            delay     = int(step.get("predicted_delay_sec", 0) or 0)

            if mode_type == "walk":
                badge = ('<span style="background:#666;color:#fff;padding:3px 7px;'
                         'border-radius:3px;font-size:11px;font-weight:bold;">WALK</span>')
                meta  = f"{dur_min} min walking"
                extra = ""
            else:
                line = step.get("line_text", "") or step.get("transport", "?")
                badge = (f'<span style="background:{color};color:#fff;padding:3px 7px;'
                         f'border-radius:3px;font-size:11px;font-weight:bold;">'
                         f'{_html.escape(str(line))}</span>')
                meta = f'{step.get("transport","").upper()} · {dur_min} min'
                if delay > 0:
                    meta += f' · +{delay}s delay'
                # headway shows the slack before missing the next connection
                hw = step_headways.get(si)
                if hw is not None:
                    hw_min = hw / 60
                    hw_col = "#c62828" if hw < 0 else ("#e65100" if hw < 120 else "#2e7d32")
                    extra = (f' &nbsp;<span style="color:{hw_col};font-size:11px;">'
                             f'({hw_min:+.1f}min slack)</span>')
                    if rob_arr:
                        extra += (f' <span style="color:#555;font-size:11px;">'
                                  f'robust arr {rob_arr}</span>')
                else:
                    extra = ""

            rows.append(
                f"""
                <tr>
                  <td style="padding:6px 8px;border-bottom:1px solid #eee;white-space:nowrap;">
                    {badge}
                  </td>
                  <td style="padding:6px 8px;border-bottom:1px solid #eee;">
                    <b>{_html.escape(from_name)}</b> &nbsp;→&nbsp; <b>{_html.escape(to_name)}</b><br>
                    <span style="color:#666;font-size:12px;">{dep} → {arr} · {meta}{extra}</span>
                  </td>
                </tr>
                """
            )

        dur_min = int(route.get("duration_sec", 0) or 0) // 60
        nt      = int(route.get("n_transfers", 0) or 0)
        walk    = float(route.get("total_walk_m", 0) or 0)
        conf    = float(route.get("route_confidence", 0) or 0) * 100
        passes  = route.get("passes_confidence")
        missed  = route.get("missed_connection")

        conf_color = "#2e7d32" if (passes is not False and conf >= 50) else "#c62828"
        conf_badge = (
            f'<span style="background:{conf_color};color:#fff;'
            f'padding:2px 9px;border-radius:4px;font-size:13px;font-weight:bold;">'
            f'{conf:.0f}%</span>'
        )
        passes_note = ""
        if passes is not None:
            passes_note = (
                f' <span style="color:{conf_color};font-size:12px;">'
                f'{"✓ passes" if passes else "✗ below"} threshold</span>'
            )
        missed_note = ""
        if missed:
            s_idx = missed.get("step_index", "?")
            hw    = float(missed.get("headway_sec", 0) or 0) / 60
            missed_note = (
                f'<div style="color:#c62828;font-size:12px;padding:2px 12px;">'
                f'⚠ Missed connection at step {s_idx} ({hw:.1f} min short)'
                f'</div>'
            )

        rob_arr_route = route.get("robust_arrival_time", "")
        rob_note = ""
        if rob_arr_route and rob_arr_route != route.get("arrival_time"):
            rob_note = (
                f'<span style="font-size:12px;color:#555;margin-left:8px;">'
                f'robust: {rob_arr_route}</span>'
            )

        header = (
            f'<div style="padding:10px 12px;">'
            f'  <div style="font-size:18px;font-weight:bold;">'
            f'    {route.get("departure_time","?")} → {route.get("arrival_time","?")}'
            f'    {rob_note}'
            f'  </div>'
            f'  <div style="margin-top:4px;font-size:13px;color:#555;">'
            f'    {dur_min} min &nbsp;·&nbsp; {nt} transfer{"s" if nt != 1 else ""}'
            f'    &nbsp;·&nbsp; {_fmt_walk(walk)} walk'
            f'    &nbsp;·&nbsp; {conf_badge} confidence{passes_note}'
            f'  </div>'
            f'</div>'
            f'{missed_note}'
        )
        table = '<table style="width:100%;border-collapse:collapse;">' + "".join(rows) + "</table>"
        return header + table
