"""
GeoSentinel - Real-Time Geo Intelligence Tracker (Streamlit dashboard).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from math import cos, radians

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import DBSCAN
from streamlit_folium import st_folium

from utils.core import MIN_FETCH_INTERVAL_SEC, DataMode, get_processed_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger("geosentinel.app")

_DEFAULT_MAP_ZOOM = 5.0
_MAP_KEY = "geosentinel_folium"
# Auto-refresh between ~30–60s (aligned with core.MIN_FETCH_INTERVAL_SEC).
_REFRESH_SEC = max(30, int(round(MIN_FETCH_INTERVAL_SEC)))
_ZOOM_SYNC_EPS = 0.08
_PAN_SYNC_EPS_DEG = 0.002

# DBSCAN: limit rows before fit to keep the UI responsive (global lat/lon, rough eps in degrees).
_CLUSTER_MAX_POINTS = 120
_DBSCAN_EPS_DEG = 0.85
_DBSCAN_MIN_SAMPLES = 2
# Rule-based insights: minimum "Hovering" labels to flag as "many".
_INSIGHT_HOVERING_MIN = 3

_DATA_MODE_OPTIONS: dict[str, str] = {
    "Auto": "auto",
    "Live Only": "live",
    "Simulated Only": "simulated",
}

_REGION_OPTIONS = [
    "Global",
    "Asia",
    "Europe",
    "North America",
    "South America",
    "Africa",
    "Oceania",
]

# (lat_min, lat_max, lon_min, lon_max)
_REGION_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "Asia": (5.0, 55.0, 60.0, 150.0),
    "Europe": (35.0, 70.0, -10.0, 40.0),
    "North America": (15.0, 70.0, -170.0, -50.0),
    "South America": (-60.0, 15.0, -90.0, -30.0),
    "Africa": (-35.0, 35.0, -20.0, 50.0),
    "Oceania": (-50.0, 0.0, 110.0, 180.0),
}

_RISK_BY_ANOMALY: dict[str, int] = {
    "High Speed": 3,
    "Low Altitude": 2,
    "Hovering": 4,
}


def _apply_custom_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #0b0f14 0%, #121826 45%, #0e1117 100%);
                color: #e8eaed;
            }
            .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1400px; }
            h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
            .geosentinel-title {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
                background: linear-gradient(90deg, #58a6ff, #79c0ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .geosentinel-sub {
                color: #8b949e;
                font-size: 0.95rem;
                margin-bottom: 1.5rem;
            }
            .intel-banner {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem 0.85rem;
                margin-bottom: 1rem;
                border-radius: 8px;
                border: 1px solid #30363d;
                background: rgba(22, 27, 34, 0.85);
                font-size: 0.82rem;
                color: #8b949e;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .intel-dot-live {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #3fb950;
                box-shadow: 0 0 10px #3fb950;
            }
            .intel-dot-cached {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #d29922;
                box-shadow: 0 0 10px #d29922;
            }
            .intel-dot-sim {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #f85149;
                box-shadow: 0 0 10px #f85149;
            }
            .intel-source-label { color: #c9d1d9; font-weight: 600; }
            .section-header {
                font-size: 1.05rem;
                font-weight: 600;
                color: #c9d1d9;
                margin: 1.25rem 0 0.75rem 0;
                padding-bottom: 0.35rem;
                border-bottom: 1px solid #30363d;
            }
            .sidebar-section {
                font-size: 0.78rem;
                font-weight: 600;
                color: #8b949e;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin: 1.1rem 0 0.5rem 0;
                padding-bottom: 0.25rem;
                border-bottom: 1px solid #21262d;
            }
            .alert-row {
                padding: 0.45rem 0.55rem;
                margin-bottom: 0.35rem;
                border-radius: 6px;
                border-left: 3px solid #f85149;
                background: rgba(248, 81, 73, 0.08);
                font-size: 0.88rem;
            }
            div[data-testid="stMetric"] {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 10px;
                padding: 0.75rem 1rem;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
                border-right: 1px solid #30363d;
            }
            .intel-insights-panel {
                margin: 1rem 0 1.25rem 0;
                padding: 1rem 1.15rem;
                border-radius: 10px;
                border: 1px solid rgba(88, 166, 255, 0.55);
                background: linear-gradient(
                    125deg,
                    rgba(88, 166, 255, 0.14) 0%,
                    rgba(22, 27, 34, 0.92) 48%,
                    rgba(121, 192, 255, 0.08) 100%
                );
                box-shadow: 0 0 28px rgba(88, 166, 255, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.04);
            }
            .intel-insight-alert {
                margin-bottom: 0.45rem;
                padding: 0.5rem 0.7rem;
                border-radius: 6px;
                border-left: 3px solid #58a6ff;
                background: rgba(88, 166, 255, 0.12);
                font-size: 0.9rem;
                color: #e6edf3;
                line-height: 1.45;
            }
            .intel-insight-alert:last-child { margin-bottom: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _intelligence_insights(
    df: pd.DataFrame,
    *,
    region_sel: str,
    region_anomaly_count: int,
) -> list[str]:
    """Lightweight, rule-based insights from the current fleet dataframe (no external AI)."""
    insights: list[str] = []
    if df.empty:
        return insights

    total_anomalies = int(df["anomaly"].notna().sum())
    if total_anomalies > 20:
        insights.append("High anomaly activity detected globally")

    if (
        region_sel != "Global"
        and total_anomalies > 0
        and region_anomaly_count > 0.30 * total_anomalies
    ):
        insights.append("High anomaly concentration detected in selected region")

    hovering_n = int((df["anomaly"] == "Hovering").sum())
    if hovering_n >= _INSIGHT_HOVERING_MIN:
        insights.append(
            "Multiple hovering aircraft detected — possible surveillance or congestion"
        )

    coords = df.dropna(subset=["lat", "lon"])
    if len(coords) >= _DBSCAN_MIN_SAMPLES:
        cluster_lbl = _dbscan_cluster_labels(coords)
        if (cluster_lbl >= 0).any():
            insights.append("Aircraft clustering detected — possible high traffic zone")

    vel = df["velocity"].dropna()
    if not vel.empty and float(vel.mean()) > 700.0:
        insights.append("High-speed movement trend observed")

    return insights


def _render_intelligence_insights_panel(
    df: pd.DataFrame,
    *,
    region_sel: str,
    region_anomaly_count: int,
) -> None:
    items = _intelligence_insights(
        df, region_sel=region_sel, region_anomaly_count=region_anomaly_count
    )
    st.markdown('<p class="section-header">AI Intelligence Insights</p>', unsafe_allow_html=True)
    if not items:
        st.markdown(
            '<div class="intel-insights-panel">'
            "<p style='margin:0;color:#8b949e;font-size:0.9rem;'>No notable patterns in the current snapshot.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return
    alerts = "".join(f'<div class="intel-insight-alert">{msg}</div>' for msg in items)
    st.markdown(
        f'<div class="intel-insights-panel">{alerts}</div>',
        unsafe_allow_html=True,
    )


def _filter_by_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if region == "Global" or df.empty:
        return df.copy() if not df.empty else df
    bounds = _REGION_BOUNDS.get(region)
    if bounds is None:
        return df.copy()
    lat_min, lat_max, lon_min, lon_max = bounds
    if "lat" not in df.columns or "lon" not in df.columns:
        return df.copy()
    mask = (
        (df["lat"] >= lat_min)
        & (df["lat"] <= lat_max)
        & (df["lon"] >= lon_min)
        & (df["lon"] <= lon_max)
    )
    return df.loc[mask].reset_index(drop=True)


def _region_center_zoom(region: str) -> tuple[float, float, float]:
    bounds = _REGION_BOUNDS[region]
    lat_min, lat_max, lon_min, lon_max = bounds
    clat = (lat_min + lat_max) / 2.0
    clon = (lon_min + lon_max) / 2.0
    span = max(abs(lat_max - lat_min), abs(lon_max - lon_min))
    if span > 80:
        z = 2.5
    elif span > 50:
        z = 3.5
    elif span > 35:
        z = 4.5
    else:
        z = 5.5
    return clat, clon, float(z)


def _sync_map_to_region_selection(region_sel: str) -> None:
    prev = st.session_state.get("_geosentinel_prev_region")
    if prev == region_sel:
        return
    st.session_state["_geosentinel_prev_region"] = region_sel
    if region_sel != "Global":
        clat, clon, zoom = _region_center_zoom(region_sel)
        st.session_state["map_center_lat"] = clat
        st.session_state["map_center_lng"] = clon
        st.session_state["map_zoom"] = zoom
    elif prev is not None and prev != "Global":
        st.session_state["_geosentinel_pending_map_reset"] = True


def _risk_scores_for_series(anomaly: pd.Series) -> pd.Series:
    def one(val: object) -> int:
        if pd.isna(val):
            return 0
        return int(_RISK_BY_ANOMALY.get(str(val), 0))

    return anomaly.map(one)


def _filter_dataframe(df: pd.DataFrame, anomalies_only: bool, speed_max: float) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["lat", "lon", "velocity"])
    out = out[out["velocity"] <= speed_max]
    if anomalies_only:
        out = out[out["anomaly"].notna()]
    return out.reset_index(drop=True)


def _dbscan_cluster_labels(df: pd.DataFrame) -> pd.Series:
    """Assign DBSCAN cluster id per row (-1 = noise). Fits on at most _CLUSTER_MAX_POINTS rows."""
    labels = pd.Series(-1, index=df.index, dtype=np.int32)
    if len(df) < _DBSCAN_MIN_SAMPLES:
        return labels
    sample = df.head(min(len(df), _CLUSTER_MAX_POINTS))
    coords = sample[["lat", "lon"]].to_numpy(dtype=np.float64)
    clustering = DBSCAN(eps=_DBSCAN_EPS_DEG, min_samples=_DBSCAN_MIN_SAMPLES).fit(coords)
    labels.loc[sample.index] = clustering.labels_.astype(np.int32)
    return labels


def _cluster_circle_radius_m(center_lat: float, deg_radius: float) -> float:
    """Approximate meters from degree radius (latitude-scaled for longitude spread)."""
    lat_rad = radians(float(center_lat))
    km_per_deg_lon = 111.32 * max(cos(lat_rad), 0.2)
    km_per_deg_lat = 110.574
    return float(max(deg_radius * max(km_per_deg_lat, km_per_deg_lon), 0.02)) * 1000.0


def _add_cluster_overlays(
    m: folium.Map,
    df: pd.DataFrame,
    labels: pd.Series,
) -> None:
    fg = folium.FeatureGroup(name="Cluster highlights", show=True)
    unique = sorted({int(x) for x in labels.unique() if x >= 0})
    for cid in unique:
        mask = labels == cid
        sub = df.loc[mask, ["lat", "lon"]]
        if sub.empty:
            continue
        c_lat = float(sub["lat"].mean())
        c_lon = float(sub["lon"].mean())
        dist_deg = float(np.hypot(sub["lat"] - c_lat, sub["lon"] - c_lon).max())
        r_m = _cluster_circle_radius_m(c_lat, max(dist_deg, 0.05))
        folium.Circle(
            location=[c_lat, c_lon],
            radius=r_m,
            color="#3fb950",
            weight=2,
            fill=True,
            fill_color="#3fb950",
            fill_opacity=0.12,
            popup=folium.Popup(f"<b>Cluster {cid}</b><br/>{int(mask.sum())} aircraft", max_width=220),
        ).add_to(fg)
    fg.add_to(m)


def _marker_color_risk(risk: int) -> str:
    if risk >= 3:
        return "red"
    if risk == 2:
        return "orange"
    return "blue"


def _build_map(
    df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    zoom_start: float = _DEFAULT_MAP_ZOOM,
    *,
    show_heatmap: bool,
    show_markers: bool,
    show_clusters: bool,
    cluster_labels: pd.Series,
) -> folium.Map:
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=int(round(zoom_start)),
        tiles="CartoDB dark_matter",
    )

    if show_heatmap and not df.empty:
        hm_data: list[list[float]] = []
        for _, row in df.iterrows():
            w = 1.0
            if pd.notna(row.get("velocity")):
                w = min(1.0 + float(row["velocity"]) / 900.0, 3.0)
            hm_data.append([float(row["lat"]), float(row["lon"]), w])
        hfg = folium.FeatureGroup(name="Heatmap (density)", show=True)
        HeatMap(hm_data, radius=18, blur=22, max_zoom=12, min_opacity=0.25).add_to(hfg)
        hfg.add_to(m)

    if show_clusters and not df.empty:
        _add_cluster_overlays(m, df, cluster_labels)

    if show_markers and not df.empty:
        mfg = folium.FeatureGroup(name="Aircraft markers", show=True)
        mc = MarkerCluster().add_to(mfg)
        for idx, row in df.iterrows():
            callsign = row["callsign"] if pd.notna(row["callsign"]) else "N/A"
            vel = row["velocity"]
            alt = row["altitude"]
            anomaly = row["anomaly"]
            is_anomaly = bool(pd.notna(anomaly))
            risk = int(row["risk_score"]) if "risk_score" in row.index and pd.notna(row.get("risk_score")) else 0
            icon_color = _marker_color_risk(risk)
            lbl = int(cluster_labels.loc[idx]) if idx in cluster_labels.index else -1
            in_cluster = lbl >= 0

            vel_s = f"{float(vel):.1f} km/h" if pd.notna(vel) else "N/A"
            alt_s = f"{float(alt):.0f} m" if pd.notna(alt) else "N/A"
            popup_lines = [
                f"<b>Callsign</b>: {callsign}",
                f"<b>Velocity</b>: {vel_s}",
                f"<b>Altitude</b>: {alt_s}",
                f"<b>Risk score</b>: {risk}",
            ]
            if is_anomaly:
                popup_lines.append(f"<b>Anomaly</b>: {anomaly}")
            if in_cluster:
                popup_lines.append(f"<b>Cluster</b>: {lbl}")
            popup_html = "<br>".join(popup_lines)
            folium.Marker(
                location=[float(row["lat"]), float(row["lon"])],
                popup=folium.Popup(popup_html, max_width=280),
                icon=folium.Icon(color=icon_color),
            ).add_to(mc)
        mfg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _default_center_from_df(df: pd.DataFrame) -> tuple[float, float]:
    return float(df["lat"].mean()), float(df["lon"].mean())


def _bounds_center(output: dict) -> tuple[float, float] | None:
    b = output.get("bounds")
    if not isinstance(b, dict):
        return None
    sw = b.get("_southWest") or {}
    ne = b.get("_northEast") or {}
    if not (all(k in sw for k in ("lat", "lng")) and all(k in ne for k in ("lat", "lng"))):
        return None
    try:
        return (float(sw["lat"]) + float(ne["lat"])) / 2, (float(sw["lng"]) + float(ne["lng"])) / 2
    except (TypeError, ValueError):
        return None


def _folium_feature_bounds_dict(m: folium.Map) -> dict | None:
    """Bounds covering all layers (same as streamlit_folium's default return value)."""
    try:
        raw = m.get_bounds()
    except (AttributeError, ValueError):
        return None
    if not raw or len(raw) != 2:
        return None
    sw, ne = raw[0], raw[1]
    if any(x is None for x in (*sw, *ne)):
        return None
    try:
        return {
            "_southWest": {"lat": float(sw[0]), "lng": float(sw[1])},
            "_northEast": {"lat": float(ne[0]), "lng": float(ne[1])},
        }
    except (TypeError, ValueError):
        return None


def _bounds_dict_matches_data_extent(out_bounds: dict, feature_bounds: dict | None, eps: float = 1e-5) -> bool:
    if feature_bounds is None:
        return False
    try:
        for corner in ("_southWest", "_northEast"):
            for k in ("lat", "lng"):
                if abs(float(out_bounds[corner][k]) - float(feature_bounds[corner][k])) > eps:
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _sync_map_view_from_st_folium(
    output: dict | None,
    sent_zoom: float,
    sent_lat: float,
    sent_lon: float,
    *,
    folium_map: folium.Map | None = None,
) -> None:
    if not output or not isinstance(output, dict):
        return

    # st_folium seeds `bounds` with m.get_bounds() (layer extent), not the live viewport. After
    # auto-refresh/remount that default is mistaken for a pan and would snap the view to the data hull.
    ob = output.get("bounds")
    if isinstance(ob, dict) and folium_map is not None:
        fb = _folium_feature_bounds_dict(folium_map)
        if _bounds_dict_matches_data_extent(ob, fb):
            return

    z_raw = output.get("zoom")
    z_out: float | None = None
    if z_raw is not None:
        try:
            z_out = float(z_raw)
        except (TypeError, ValueError):
            z_out = None

    sent_zoom = float(sent_zoom)
    zoom_changed = z_out is not None and abs(z_out - sent_zoom) > _ZOOM_SYNC_EPS
    if zoom_changed and z_out is not None:
        st.session_state["map_zoom"] = z_out
        logger.info("Map view sync (zoom): %s", z_out)

    bc = _bounds_center(output)
    if bc is not None:
        clat, clng = bc
        if (
            abs(clat - sent_lat) > _PAN_SYNC_EPS_DEG
            or abs(clng - sent_lon) > _PAN_SYNC_EPS_DEG
        ):
            st.session_state["map_center_lat"] = clat
            st.session_state["map_center_lng"] = clng
            logger.info("Map view sync (bounds center): (%s, %s)", clat, clng)


def _map_view_for_render(
    default_lat: float,
    default_lon: float,
) -> tuple[float, float, float]:
    zoom = float(st.session_state.get("map_zoom", _DEFAULT_MAP_ZOOM))
    lat = st.session_state.get("map_center_lat")
    lon = st.session_state.get("map_center_lng")
    if lat is None or lon is None:
        return default_lat, default_lon, zoom
    try:
        return float(lat), float(lon), zoom
    except (TypeError, ValueError):
        return default_lat, default_lon, zoom


def _ui_source_display(fetch_status: str) -> tuple[str, str]:
    """
    Map backend status to banner label and CSS dot class.
    Live = green, Cached = yellow, Simulated = red.
    """
    if fetch_status == "simulated":
        return "Simulated", "intel-dot-sim"
    if fetch_status in ("ok",):
        return "Live", "intel-dot-live"
    if fetch_status in ("cached", "rate_limited_cached", "session_cached"):
        return "Cached", "intel-dot-cached"
    # Degraded / unknown but still showing a frame: treat as cached-style fallback.
    return "Cached", "intel-dot-cached"


def _render_live_alerts(df: pd.DataFrame) -> None:
    slot = st.session_state.get("_live_alerts_slot")
    if slot is None:
        return
    anomalies = df[df["anomaly"].notna()][["callsign", "anomaly"]].copy()
    with slot.container():
        if anomalies.empty:
            st.caption("No active anomalies.")
        else:
            for _, r in anomalies.iterrows():
                cs = r["callsign"] if pd.notna(r["callsign"]) else "N/A"
                st.markdown(
                    f'<div class="alert-row"><b>{cs}</b><br/><span style="color:#8b949e">{r["anomaly"]}</span></div>',
                    unsafe_allow_html=True,
                )


def _append_temporal_snapshot(aircraft_count: int, anomaly_count: int) -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(
        {
            "timestamp": datetime.now(timezone.utc),
            "aircraft_count": aircraft_count,
            "anomaly_count": anomaly_count,
        }
    )
    if len(st.session_state.history) > 5:
        st.session_state.history = st.session_state.history[-5:]


@st.fragment(run_every=_REFRESH_SEC)
def _render_dashboard() -> None:
    _m = st.session_state.get("geosentinel_mode", "auto")
    mode: DataMode = _m if _m in ("auto", "live", "simulated") else "auto"
    logger.info("Dashboard fragment run: mode=%s", mode)

    region_sel = st.session_state.get("region_filter", "Global")
    if region_sel not in _REGION_OPTIONS:
        region_sel = "Global"

    try:
        df, fetch_status = get_processed_data(mode=mode)
    except Exception:
        logger.exception("get_processed_data() raised an exception")
        st.warning("Unable to reach the aircraft data pipeline. Showing last known data if available.")
        df = pd.DataFrame()
        fetch_status = "error"

    if df.empty:
        cached = st.session_state.get("cached_aircraft_df")
        if cached is not None and not getattr(cached, "empty", True):
            df = cached.copy()
            fetch_status = "session_cached"
            logger.warning("Using session-cached aircraft data (last fetch had no rows)")
        else:
            msg = (
                "OpenSky rate limited your connection (HTTP 429). Try Auto mode or wait before refreshing."
                if fetch_status == "rate_limited"
                else "Unable to load aircraft data. The source may be unavailable or returned no valid records."
            )
            logger.warning("No aircraft data available (status=%s)", fetch_status)
            st.warning(msg)
            return

    if not df.empty:
        st.session_state["cached_aircraft_df"] = df.copy()

    total_aircraft = int(len(df))
    total_anomalies = int(df["anomaly"].notna().sum())
    _append_temporal_snapshot(total_aircraft, total_anomalies)

    label, dot_class = _ui_source_display(fetch_status)
    st.markdown(
        f'<div class="intel-banner"><span class="{dot_class}"></span>'
        f'<span>Operations console</span>'
        f'<span style="margin-left:auto" class="intel-source-label">Data Source: {label}</span></div>',
        unsafe_allow_html=True,
    )

    if fetch_status == "simulated":
        st.info("Running on simulated intelligence data.")

    if mode == "live" and fetch_status in ("rate_limited", "unavailable", "empty", "error"):
        st.warning(
            "Live Only mode could not refresh from the API. Showing the last successful snapshot for this session."
        )

    if fetch_status == "rate_limited_cached":
        st.warning(
            "OpenSky returned HTTP **429 (Too Many Requests)**. Showing the last successful snapshot; "
            "data may be slightly stale. Wait before refreshing or use authenticated OpenSky access for higher limits."
        )
    elif fetch_status == "cached":
        logger.info("Serving data from TTL cache or disk snapshot (OpenSky-friendly pacing)")
    elif fetch_status == "session_cached":
        st.warning(
            "Live fetch failed or was empty; showing the last successful snapshot stored for this session."
        )

    _render_live_alerts(df)

    show_heatmap = bool(st.session_state.get("show_heatmap", False))
    show_clusters = bool(st.session_state.get("show_clusters", False))
    show_markers = bool(st.session_state.get("show_markers", True))

    anomalies_only = bool(st.session_state.get("show_anomalies_only", False))
    speed_threshold = float(st.session_state.get("speed_threshold", 1200))

    logger.info(
        "Data OK: total_aircraft=%s total_anomalies=%s filters=(anomalies_only=%s speed_max=%s)",
        total_aircraft,
        total_anomalies,
        anomalies_only,
        speed_threshold,
    )

    st.markdown('<p class="section-header">Fleet Overview</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric("Total Aircraft", f"{total_aircraft:,}")
    with c2:
        st.metric("Total Anomalies", f"{total_anomalies:,}")
    with c3:
        st.metric("Refresh cadence", f"{_REFRESH_SEC}s", help="Balanced for OpenSky rate limits (≈30–60s)")

    df_region = _filter_by_region(df, region_sel)
    region_aircraft = int(len(df_region))
    region_anomalies = int(df_region["anomaly"].notna().sum())

    _render_intelligence_insights_panel(
        df, region_sel=region_sel, region_anomaly_count=region_anomalies
    )
    anom_series = df_region.loc[df_region["anomaly"].notna(), "anomaly"]
    if anom_series.empty:
        most_common_type = "—"
    else:
        mode_vals = anom_series.mode()
        most_common_type = str(mode_vals.iloc[0]) if len(mode_vals) else "—"

    st.markdown('<p class="section-header">Regional Intelligence</p>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Aircraft in region", f"{region_aircraft:,}")
    with r2:
        st.metric("Anomalies in region", f"{region_anomalies:,}")
    with r3:
        st.metric("Most common anomaly (region)", most_common_type)

    st.markdown("**Regional Intelligence Summary**")
    if region_sel == "Global":
        st.caption(
            "The operational picture covers **all** reported aircraft. Use the map and layers to scan density, "
            "clusters, and marker risk coloring (blue = baseline, orange = medium risk, red = high risk)."
        )
    else:
        b = _REGION_BOUNDS.get(region_sel)
        b_txt = (
            f"lat [{b[0]:.0f}°, {b[1]:.0f}°], lon [{b[2]:.0f}°, {b[3]:.0f}°]"
            if b
            else ""
        )
        st.caption(
            f"**{region_sel}** filter is active ({b_txt}). "
            f"There are **{region_aircraft:,}** aircraft and **{region_anomalies:,}** anomalies in this box. "
            f"The dominant anomaly label is **{most_common_type}**."
        )

    df_region = df_region.copy()
    df_region["risk_score"] = _risk_scores_for_series(df_region["anomaly"])
    filtered = _filter_dataframe(df_region, anomalies_only, speed_threshold)

    if "risk_score" not in filtered.columns and not filtered.empty:
        filtered = filtered.copy()
        filtered["risk_score"] = _risk_scores_for_series(filtered["anomaly"])

    # Limit work: only cluster when layers need it; cap points inside _dbscan_cluster_labels.
    if show_clusters or show_markers:
        cluster_labels = _dbscan_cluster_labels(filtered) if not filtered.empty else pd.Series(dtype=np.int32)
    else:
        cluster_labels = pd.Series(dtype=np.int32)

    if filtered.empty:
        logger.warning("No rows after filters (anomalies_only=%s speed_max=%s)", anomalies_only, speed_threshold)
        st.warning("No aircraft match the current filters. Adjust the sidebar controls or wait for the next refresh.")
        def_lat, def_lon = _default_center_from_df(df) if not df.empty else (20.0, 0.0)
        cluster_labels = pd.Series(dtype=np.int32)
    else:
        def_lat, def_lon = _default_center_from_df(filtered)

    if st.session_state.pop("_geosentinel_pending_map_reset", False):
        st.session_state["map_center_lat"] = def_lat
        st.session_state["map_center_lng"] = def_lon
        st.session_state["map_zoom"] = _DEFAULT_MAP_ZOOM

    if "map_zoom" not in st.session_state:
        st.session_state["map_zoom"] = _DEFAULT_MAP_ZOOM
    if "map_center_lat" not in st.session_state or "map_center_lng" not in st.session_state:
        st.session_state["map_center_lat"] = def_lat
        st.session_state["map_center_lng"] = def_lon

    center_lat, center_lon, map_zoom = _map_view_for_render(def_lat, def_lon)

    labels_for_map = (
        cluster_labels
        if not cluster_labels.empty
        else pd.Series(-1, index=filtered.index, dtype=np.int32)
    )

    m = _build_map(
        filtered,
        center_lat,
        center_lon,
        zoom_start=map_zoom,
        show_heatmap=show_heatmap,
        show_markers=show_markers,
        show_clusters=show_clusters,
        cluster_labels=labels_for_map,
    )

    output = st_folium(
        m,
        height=560,
        width=None,
        use_container_width=True,
        returned_objects=["zoom", "bounds"],
        zoom=map_zoom,
        center=(center_lat, center_lon),
        key=_MAP_KEY,
    )
    _sync_map_view_from_st_folium(
        output if isinstance(output, dict) else None,
        sent_zoom=map_zoom,
        sent_lat=center_lat,
        sent_lon=center_lon,
        folium_map=m,
    )

    st.caption(
        f"Showing **{len(filtered)}** aircraft · Region **{region_sel}** · Heatmap: **{'on' if show_heatmap else 'off'}** · "
        f"Clusters: **{'on' if show_clusters else 'off'}** · Markers: **{'on' if show_markers else 'off'}** · "
        f"Auto-refresh **{_REFRESH_SEC}s** · Speed cap **{speed_threshold:.0f} km/h** · Mode **{mode}** · "
        f"Markers: blue (normal), orange (risk 2), red (risk ≥3)"
    )
    logger.info("Map render complete; markers=%s", len(filtered))

    st.markdown('<p class="section-header">Temporal Intelligence</p>', unsafe_allow_html=True)
    hist = st.session_state.get("history") or []
    if len(hist) >= 2:
        hist_df = pd.DataFrame(hist)
        hist_df = hist_df.sort_values("timestamp")
        ts_col = pd.to_datetime(hist_df["timestamp"])
        chart_base = pd.DataFrame(
            {
                "Time": ts_col.dt.tz_convert(None),
                "Aircraft count": hist_df["aircraft_count"].values,
                "Anomaly count": hist_df["anomaly_count"].values,
            }
        )
        st.caption("Trend from the last snapshots (auto-refresh).")
        ac_chart = chart_base.set_index("Time")[["Aircraft count"]]
        an_chart = chart_base.set_index("Time")[["Anomaly count"]]
        st.line_chart(ac_chart, height=220)
        st.line_chart(an_chart, height=220)
    else:
        st.caption("Temporal charts appear after at least two refresh cycles.")

    st.markdown('<p class="section-header">Threat Intelligence</p>', unsafe_allow_html=True)
    if filtered.empty:
        st.caption("No aircraft match the current filters for threat ranking.")
        top_threats = pd.DataFrame(columns=["callsign", "anomaly", "risk_score"])
    else:
        sort_df = filtered.sort_values("risk_score", ascending=False)
        top_threats = sort_df[["callsign", "anomaly", "risk_score"]].head(5).copy()
        top_threats["callsign"] = top_threats["callsign"].fillna("N/A")
        top_threats["anomaly"] = top_threats["anomaly"].fillna("—")

    st.markdown("**Threat Intelligence Panel** *(top 5 by risk score)*")
    st.dataframe(
        top_threats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_score": st.column_config.NumberColumn("Risk score", format="%d"),
        },
    )


def main() -> None:
    st.set_page_config(
        page_title="GeoSentinel - Real-Time Geo Intelligence Tracker",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_custom_css()

    st.markdown(
        '<p class="geosentinel-title">GeoSentinel — Geo Intelligence Dashboard</p>'
        '<p class="geosentinel-sub">OpenSky live states · Cache & simulation fallback · Heatmap · DBSCAN · Anomaly alerts</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<p class="sidebar-section">Data source</p>', unsafe_allow_html=True)
        mode_label = st.selectbox(
            "Data Source Mode",
            list(_DATA_MODE_OPTIONS.keys()),
            index=0,
            help="Auto uses API with cache + simulation fallback. Live Only never falls back to simulation.",
        )
        st.session_state["geosentinel_mode"] = _DATA_MODE_OPTIONS[mode_label]

        st.markdown('<p class="sidebar-section">Regional filter</p>', unsafe_allow_html=True)
        st.selectbox(
            "Region Filter",
            _REGION_OPTIONS,
            index=0,
            key="region_filter",
        )
        _sync_map_to_region_selection(st.session_state.get("region_filter", "Global"))

        st.markdown('<p class="sidebar-section">Mission controls</p>', unsafe_allow_html=True)
        st.checkbox("Show Anomalies Only", key="show_anomalies_only")
        st.slider(
            "Speed Threshold",
            min_value=100,
            max_value=1200,
            value=1200,
            step=10,
            key="speed_threshold",
            help="Show aircraft with ground speed at or below this value (km/h).",
        )
        st.markdown('<p class="sidebar-section">Map layers</p>', unsafe_allow_html=True)
        st.checkbox("Show Heatmap", key="show_heatmap", help="Aircraft density (folium.plugins.HeatMap)")
        st.checkbox("Show Clusters", key="show_clusters", help="DBSCAN clusters (lat/lon) + green highlights")
        st.checkbox(
            "Show Markers",
            value=True,
            key="show_markers",
            help="Risk-based colors: blue · orange (medium) · red (high)",
        )
        st.markdown('<p class="sidebar-section">Live Alerts</p>', unsafe_allow_html=True)
        st.session_state["_live_alerts_slot"] = st.empty()

    _render_dashboard()


if __name__ == "__main__":
    main()
