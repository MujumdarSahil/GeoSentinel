"""
OpenSky aircraft data: fetch, clean, and anomaly detection for real-time tracking.

Supports live API, on-disk cache (rate-limit friendly), and simulated fallback.
"""

from __future__ import annotations

import json
import logging
import os
import random
import string
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
# If last successful fetch is younger than this, prefer cache over a new HTTP call (reduces 429s).
CACHE_TTL_SEC = 60.0
MIN_FETCH_INTERVAL_SEC = 45.0

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = _PROJECT_ROOT / "data" / "cache.json"

# State vector indices (OpenSky API)
IDX_CALLSIGN = 1
IDX_LONGITUDE = 5
IDX_LATITUDE = 6
IDX_BARO_ALTITUDE = 7
IDX_VELOCITY = 9

# Processing limits
MAX_ROWS = 200
MS_TO_KMH = 3.6

DataMode = Literal["auto", "live", "simulated"]


def _empty_result() -> pd.DataFrame:
    """Schema-aligned empty frame for failed fetches."""
    return pd.DataFrame(
        columns=["callsign", "lat", "lon", "velocity", "altitude", "anomaly"]
    )


def save_cache(data: dict[str, Any]) -> None:
    """
    Persist the raw OpenSky API JSON payload plus a wall-clock timestamp for TTL and fallback.
    Uses an atomic replace where supported.
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    envelope: dict[str, Any] = {"timestamp": time.time(), "payload": data}
    fd, tmp_path = tempfile.mkstemp(dir=CACHE_PATH.parent, suffix=".tmp")
    tmp_file = Path(tmp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(envelope, f, separators=(",", ":"))
        os.replace(tmp_file, CACHE_PATH)
    except Exception:
        try:
            tmp_file.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_cache() -> dict[str, Any] | None:
    """
    Read cache from disk. Returns {"timestamp": float, "payload": dict} or None if missing/invalid.
    """
    if not CACHE_PATH.is_file():
        return None
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            envelope = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("Cache file unreadable; ignoring: %s", e)
        return None
    if not isinstance(envelope, dict):
        return None
    ts = envelope.get("timestamp")
    payload = envelope.get("payload")
    if not isinstance(ts, (int, float)) or not isinstance(payload, dict):
        return None
    return {"timestamp": float(ts), "payload": payload}


def generate_simulated_data() -> pd.DataFrame:
    """
    Build 100–200 synthetic aircraft rows (for demos / API outage).
    Velocities and altitudes are chosen so anomaly rules can still fire sometimes.
    """
    rng = random.Random()
    n = rng.randint(100, 200)
    callsign_chars = string.ascii_uppercase + string.digits

    rows: list[dict[str, Any]] = []
    for _ in range(n):
        cs = "".join(rng.choices(callsign_chars, k=rng.randint(4, 7)))
        lat = rng.uniform(-90.0, 90.0)
        lon = rng.uniform(-180.0, 180.0)
        velocity = rng.uniform(200.0, 950.0)
        altitude = rng.uniform(500.0, 12000.0)
        # Sprinkle a few obvious anomalies so the dashboard is not empty in demo mode.
        if rng.random() < 0.04:
            velocity = rng.uniform(920.0, 980.0)
        if rng.random() < 0.04:
            altitude = rng.uniform(200.0, 900.0)
        if rng.random() < 0.02:
            velocity = rng.uniform(0.0, 45.0)
            altitude = rng.uniform(3500.0, 9000.0)

        rows.append(
            {
                "callsign": cs,
                "lat": lat,
                "lon": lon,
                "velocity": velocity,
                "altitude": altitude,
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["lat", "lon"])
    df = df.head(MAX_ROWS).reset_index(drop=True)
    return _add_anomaly_column(df)


def _fetch_opensky_states() -> tuple[dict[str, Any] | None, int | None]:
    """
    GET OpenSky states/all.
    Returns (payload, None) on success, or (None, http_status_or_None) on failure.
    """
    try:
        response = requests.get(
            OPENSKY_STATES_URL,
            timeout=30,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.Timeout as e:
        logger.error("OpenSky API request timed out: %s", e)
        return None, None
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else None
        if code == 429:
            logger.warning(
                "OpenSky API rate limited (429). Wait before retrying; use longer refresh intervals."
            )
        else:
            logger.error("OpenSky API HTTP error: %s", e)
        return None, code
    except requests.exceptions.RequestException as e:
        logger.error("OpenSky API request failed: %s", e)
        return None, None
    except ValueError as e:
        logger.error("OpenSky API returned invalid JSON: %s", e)
        return None, None


def _states_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Parse OpenSky payload into a raw DataFrame with expected columns.
    Missing or short state vectors are handled safely.
    """
    states = payload.get("states")
    if not states:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    max_idx = max(IDX_CALLSIGN, IDX_LONGITUDE, IDX_LATITUDE, IDX_BARO_ALTITUDE, IDX_VELOCITY)

    for state in states:
        if not isinstance(state, (list, tuple)) or len(state) <= max_idx:
            continue
        rows.append(
            {
                "callsign": state[IDX_CALLSIGN],
                "lon": state[IDX_LONGITUDE],
                "lat": state[IDX_LATITUDE],
                "baro_altitude": state[IDX_BARO_ALTITUDE],
                "velocity_ms": state[IDX_VELOCITY],
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ("lon", "lat", "baro_altitude", "velocity_ms"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def _clean_callsign(val: Any) -> Any:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return pd.NA
        s = str(val).strip()
        return s if s else pd.NA

    df["callsign"] = df["callsign"].apply(_clean_callsign)

    return df


def _process_aircraft_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid positions, convert units, rename, filter altitude, cap rows."""
    if df.empty:
        return df

    out = df.dropna(subset=["lat", "lon"]).copy()
    out["velocity"] = out["velocity_ms"] * MS_TO_KMH
    out = out.rename(columns={"baro_altitude": "altitude"})
    out = out.drop(columns=["velocity_ms"], errors="ignore")
    out = out[["callsign", "lat", "lon", "velocity", "altitude"]]

    out = out[out["altitude"] > 0]
    out = out.head(MAX_ROWS)
    return out.reset_index(drop=True)


def _classify_anomaly(row: pd.Series) -> str | None:
    """
    Single anomaly label per row (priority: high speed, then low altitude, then hover).
    """
    velocity = row["velocity"]
    altitude = row["altitude"]

    if pd.isna(velocity) or pd.isna(altitude):
        return None

    if velocity > 900:
        return "High Speed"
    if altitude < 1000:
        return "Low Altitude"
    if velocity < 50 and altitude > 3000:
        return "Hovering"
    return None


def _add_anomaly_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["anomaly"] = pd.Series(dtype=object)
        return out
    out["anomaly"] = out.apply(_classify_anomaly, axis=1)
    return out


def _dataframe_from_api_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """states → clean → anomalies (same pipeline as live fetch)."""
    raw = _states_to_dataframe(payload)
    processed = _process_aircraft_data(raw)
    if processed.empty:
        return _empty_result()
    return _add_anomaly_column(processed)


def _try_disk_cache_to_dataframe(http_code: int | None) -> tuple[pd.DataFrame, str]:
    """Parse on-disk cache into a DataFrame; never raises for bad cache."""
    envelope = load_cache()
    if envelope is None:
        if http_code == 429:
            return _empty_result(), "rate_limited"
        return _empty_result(), "unavailable"

    try:
        df = _dataframe_from_api_payload(envelope["payload"])
    except Exception as e:
        logger.exception("Unexpected error while processing cached OpenSky data: %s", e)
        if http_code == 429:
            return _empty_result(), "rate_limited"
        return _empty_result(), "unavailable"

    if df.empty:
        if http_code == 429:
            return _empty_result(), "rate_limited"
        return _empty_result(), "empty"

    if http_code == 429:
        return df, "rate_limited_cached"
    return df, "cached"


def get_processed_data(mode: DataMode = "auto") -> tuple[pd.DataFrame, str]:
    """
    Fetch → clean → detect anomalies, with mode-specific sourcing.

    Modes
    -------
    - ``auto``: prefer fresh cache if younger than CACHE_TTL_SEC; else call API; on HTTP failure
      (including 429) fall back to disk cache; if cache is unusable, use simulated data.
    - ``live``: OpenSky HTTP only; no TTL cache short-circuit, no disk fallback, no simulation.
    - ``simulated``: synthetic fleet only (still runs anomaly detection).

    Returns (dataframe, status):
      - ``ok`` — fresh HTTP response processed successfully
      - ``cached`` — served from on-disk cache within TTL, or stale cache after non-429 failure (auto)
      - ``rate_limited`` — HTTP 429 (or other failure) and no usable cache (live/auto without sim path)
      - ``rate_limited_cached`` — HTTP 429; served from on-disk cache
      - ``unavailable`` — other failure and no usable cache
      - ``empty`` — API or cache produced no usable rows after cleaning
      - ``error`` — unexpected processing error on a fresh successful HTTP body
      - ``simulated`` — synthetic dataset
    """
    if mode == "simulated":
        logger.info("Using Simulated Data")
        df = generate_simulated_data()
        if df.empty:
            logger.warning("Simulated generator returned no rows")
            return _empty_result(), "empty"
        logger.info("Simulated pipeline OK: %s aircraft rows", len(df))
        return df, "simulated"

    now = time.time()

    # Auto: within TTL, skip HTTP and use cache (reduces rate limits).
    if mode == "auto":
        cached = load_cache()
        if cached is not None and (now - cached["timestamp"]) < CACHE_TTL_SEC:
            logger.info("Using Cached Data")
            try:
                df = _dataframe_from_api_payload(cached["payload"])
            except Exception as e:
                logger.exception("Unexpected error while processing cached OpenSky data: %s", e)
                df_sim = generate_simulated_data()
                if not df_sim.empty:
                    logger.info("Using Simulated Data")
                    return df_sim, "simulated"
                return _empty_result(), "error"
            if df.empty:
                logger.warning("OpenSky: no rows after cleaning/limits (from cache)")
                df_sim = generate_simulated_data()
                if not df_sim.empty:
                    logger.info("Using Simulated Data")
                    return df_sim, "simulated"
                return _empty_result(), "empty"
            logger.info("OpenSky pipeline OK (cache TTL): %s aircraft rows", len(df))
            return df, "cached"

    logger.info("Fetching new data")
    payload, http_code = _fetch_opensky_states()

    if payload is None:
        if mode == "live":
            logger.warning("Live mode: API failed (no fallback)")
            if http_code == 429:
                return _empty_result(), "rate_limited"
            return _empty_result(), "unavailable"

        logger.info("API failed, attempting disk cache then simulation")
        df, st = _try_disk_cache_to_dataframe(http_code)
        if not df.empty:
            logger.info("Using Cached Data")
            return df, st

        df_sim = generate_simulated_data()
        if not df_sim.empty:
            logger.info("Using Simulated Data")
            return df_sim, "simulated"
        return _empty_result(), st if st in ("rate_limited", "unavailable", "empty") else "unavailable"

    try:
        df = _dataframe_from_api_payload(payload)
    except Exception as e:
        logger.exception("Unexpected error while processing OpenSky data: %s", e)
        if mode == "live":
            return _empty_result(), "error"
        df_disk, st = _try_disk_cache_to_dataframe(http_code)
        if not df_disk.empty:
            logger.info("Using Cached Data")
            return df_disk, st
        df_sim = generate_simulated_data()
        if not df_sim.empty:
            logger.info("Using Simulated Data")
            return df_sim, "simulated"
        return _empty_result(), "error"

    try:
        save_cache(payload)
    except OSError as e:
        logger.error("Could not write cache file: %s", e)

    if df.empty:
        logger.warning("OpenSky: no rows after cleaning/limits")
        if mode == "live":
            return _empty_result(), "empty"
        df_disk, st = _try_disk_cache_to_dataframe(http_code)
        if not df_disk.empty:
            logger.info("Using Cached Data")
            return df_disk, st
        df_sim = generate_simulated_data()
        if not df_sim.empty:
            logger.info("Using Simulated Data")
            return df_sim, "simulated"
        return _empty_result(), "empty"

    logger.info("Using Live Data")
    logger.info("OpenSky pipeline OK: %s aircraft rows", len(df))
    return df, "ok"
