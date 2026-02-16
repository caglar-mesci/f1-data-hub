import os
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# F1 DATA HUB — Streamlit Dashboard
#
# Purpose
# - Explore Formula 1 timing data via FastF1
# - Compare two drivers: pace, delta, and tyre strategy (stints)
# - Provide a clean UX: show friendly messages when data is unavailable
#
# Design principles
# - Single-file app for easy deployment and portability
# - Strong inline documentation for maintainability
# - Defensive programming: avoid exposing raw exceptions to end users
#
# Streamlit note
# - Streamlit re-runs this script top-to-bottom on every interaction.
# - We use st.session_state to keep a lightweight "page" navigation state.
# =============================================================================

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="F1 Data Hub", layout="wide")

# =============================================================================
# Caching: FastF1 local cache
# =============================================================================
def ensure_fastf1_cache() -> None:
    """
    Create and enable a local cache folder for FastF1.

    Why:
    - FastF1 downloads timing resources during session.load()
    - Caching prevents repeated downloads and improves responsiveness
    """
    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache("cache")


# =============================================================================
# Formatting helpers
# =============================================================================
def fmt_timedelta(td: pd.Timedelta) -> str:
    """
    Convert a pandas Timedelta to a racing-style string:
      mm:ss.mmm (or h:mm:ss.mmm when needed)

    This avoids verbose displays like:
      '0 days 00:01:23.456000'
    """
    if pd.isna(td):
        return "N/A"

    total_ms = int(td.total_seconds() * 1000)
    sign = "-" if total_ms < 0 else ""
    total_ms = abs(total_ms)

    ms = total_ms % 1000
    total_s = total_ms // 1000

    s = total_s % 60
    total_m = total_s // 60

    m = total_m % 60
    h = total_m // 60

    if h > 0:
        return f"{sign}{h}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{sign}{m}:{s:02d}.{ms:03d}"


def td_series_to_seconds(td_series: pd.Series) -> pd.Series:
    """
    Matplotlib cannot directly plot Timedelta values.
    Convert Timedelta -> float seconds for plotting.
    """
    return td_series.dt.total_seconds()


def safe_driver_fullname(session, abbr: str) -> str:
    """
    Map a driver abbreviation (e.g., 'VER') to a full name.

    If metadata is missing or lookup fails, fall back to the abbreviation.
    """
    try:
        return session.get_driver(abbr)["FullName"]
    except Exception:
        return abbr


# =============================================================================
# Cached data loaders (Streamlit cache)
# =============================================================================
@st.cache_data(show_spinner=False)
def get_supported_years() -> List[int]:
    """
    Discover seasons that have a non-empty event schedule.

    This scan is cached because it is expensive and does not need to be repeated
    on every UI interaction.
    """
    years: List[int] = []
    for y in range(1950, 2026):
        try:
            sched = fastf1.get_event_schedule(y)
            if sched is not None and len(sched) > 0:
                years.append(y)
        except Exception:
            # Ignore years that are unavailable in the provider backend.
            pass
    return years


@st.cache_data(show_spinner=False)
def get_year_schedule(year: int) -> pd.DataFrame:
    """Return the event schedule for a specific season year (cached)."""
    return fastf1.get_event_schedule(year)


@st.cache_data(show_spinner=False)
def safe_load_session(year: int, event_name: str, session_code: str):
    """
    Load a FastF1 session safely.

    Returns:
      (session_object, error_message)

    - If load succeeds and laps are available: (session, None)
    - If session doesn't exist or data is missing: (None, friendly_message)

    This function intentionally hides raw exceptions from the UI.
    """
    try:
        ensure_fastf1_cache()
        s = fastf1.get_session(year, event_name, session_code)
        s.load()

        # Even if load succeeds, some sessions may have no lap data.
        if s.laps is None or s.laps.empty:
            return None, (
                f"No lap data found for **{year} {event_name} ({session_code})**. "
                "This session may not exist for this event, or data may not be available."
            )

        return s, None

    except Exception:
        return None, (
            f"No data available for **{year} {event_name} ({session_code})**. "
            "Try a different session (e.g., Race or Qualifying) or select another event."
        )


# =============================================================================
# Lap filtering and KPIs
# =============================================================================
def filter_by_lap_range(laps: pd.DataFrame, lap_range: Tuple[int, int]) -> pd.DataFrame:
    """Filter laps by inclusive LapNumber range."""
    if laps.empty or "LapNumber" not in laps.columns:
        return laps
    start, end = lap_range
    return laps[(laps["LapNumber"] >= start) & (laps["LapNumber"] <= end)]


def laps_for_pace(laps: pd.DataFrame, quicklaps_only: bool, lap_range: Tuple[int, int]) -> pd.DataFrame:
    """
    Prepare laps for pace and delta charts.

    quicklaps_only:
    - Removes in/out laps and obvious outliers
    - Produces cleaner pace plots

    Note:
    - This can remove laps around pit cycles/restarts.
      Therefore it should not be used for tyre stint detection.
    """
    laps = filter_by_lap_range(laps, lap_range)
    if quicklaps_only:
        laps = laps.pick_quicklaps()
    return laps


def laps_for_strategy(laps: pd.DataFrame, lap_range: Tuple[int, int]) -> pd.DataFrame:
    """
    Prepare laps for tyre strategy analysis.

    Strategy/stints require the full lap sequence (no quicklap filtering),
    otherwise compound changes might be missed.
    """
    laps = filter_by_lap_range(laps, lap_range).sort_values("LapNumber").copy()

    # Compound can occasionally be missing for a few rows depending on provider.
    # Forward-fill helps preserve continuity for stint detection.
    if "Compound" in laps.columns:
        laps["Compound"] = laps["Compound"].ffill()

    return laps


def compute_kpis(laps: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic KPIs for the given lap set."""
    lt = laps["LapTime"].dropna()
    avg = lt.mean() if not lt.empty else pd.NaT
    best = lt.min() if not lt.empty else pd.NaT

    pits = 0
    if "PitInTime" in laps.columns:
        pits = int(laps["PitInTime"].notna().sum())

    lap_count = int(laps["LapNumber"].nunique()) if "LapNumber" in laps.columns else 0
    return {"avg": avg, "best": best, "pits": pits, "laps": lap_count}


# =============================================================================
# Tyre stint detection
# =============================================================================
def build_stints(laps: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect tyre stints from lap data.

    A stint = consecutive laps on the same compound.

    Output format:
    [
      {
        "compound": "HARD",
        "start_lap": 15,
        "end_lap": 53,
        "length": 39,
        "start_tyre_life": 1,
        "end_tyre_life": 39
      },
      ...
    ]
    """
    if laps.empty or "Compound" not in laps.columns:
        return []

    df = laps.sort_values("LapNumber").copy()
    if df["Compound"].isna().all():
        return []

    stints: List[Dict[str, Any]] = []

    current_comp = None
    start_lap = None
    start_life = None
    last_lap = None
    last_life = None

    for _, row in df.iterrows():
        comp = row.get("Compound", None)
        if pd.isna(comp):
            continue

        lapn = int(row["LapNumber"])
        life = row.get("TyreLife", np.nan)

        # Initialize the first stint
        if current_comp is None:
            current_comp = comp
            start_lap = lapn
            start_life = life
            last_lap = lapn
            last_life = life
            continue

        # Close previous stint when compound changes
        if comp != current_comp:
            stints.append({
                "compound": current_comp,
                "start_lap": start_lap,
                "end_lap": last_lap,
                "length": last_lap - start_lap + 1,
                "start_tyre_life": start_life,
                "end_tyre_life": last_life,
            })
            current_comp = comp
            start_lap = lapn
            start_life = life

        # Track last seen lap for current stint
        last_lap = lapn
        last_life = life

    # Close final stint
    if current_comp is not None and start_lap is not None and last_lap is not None:
        stints.append({
            "compound": current_comp,
            "start_lap": start_lap,
            "end_lap": last_lap,
            "length": last_lap - start_lap + 1,
            "start_tyre_life": start_life,
            "end_tyre_life": last_life,
        })

    return stints


# =============================================================================
# Simple page navigation
# =============================================================================
if "page" not in st.session_state:
    st.session_state.page = "homepage"


def goto(page_name: str) -> None:
    """
    Update the current page value in session_state.
    Navigation buttons should call st.rerun() to apply immediately.
    """
    st.session_state.page = page_name


# =============================================================================
# HOMEPAGE
# =============================================================================
if st.session_state.page == "homepage":
    st.title("🏁 F1 Data Hub")

    st.markdown(
        """
### Formula 1 Data Exploration & Comparison Platform

Use this dashboard to:
- Select **Season → Grand Prix → Session**
- Compare **two drivers** using pace and delta charts
- Review **tyre strategy** with automatic stint detection

Data source: FastF1 timing data.
"""
    )

    st.subheader("🏆 Most World Championships")
    champions = [
        ("Michael Schumacher", 7),
        ("Lewis Hamilton", 7),
        ("Juan Manuel Fangio", 5),
        ("Alain Prost", 4),
        ("Sebastian Vettel", 4),
        ("Max Verstappen", 4),
    ]
    for name, titles in champions:
        st.write(f"**{titles}×** — {name}")

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Enter Platform →", use_container_width=True):
            goto("explore")
            st.rerun()  # Ensures navigation works on the first click
    with c2:
        st.info("First load may take a few seconds. Later runs are faster thanks to caching.")

    st.divider()
    st.markdown(
        """
**Built by Çağlar Mesci**  
GitHub: https://github.com/caglar-mesci
"""
    )
    st.stop()


# =============================================================================
# EXPLORE PAGE
# =============================================================================
st.title("🏎️ Explore & Compare")

years = get_supported_years()
if not years:
    st.warning("No supported seasons found from the data provider.")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar: user controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Navigation")
    if st.button("🏠 Homepage", use_container_width=True):
        goto("homepage")
        st.rerun()

    st.divider()
    st.header("Race selection")

    default_year = 2021 if 2021 in years else max(years)
    year = st.selectbox("Season", years, index=years.index(default_year))

    schedule = get_year_schedule(year)

    # The schedule DataFrame can have different column names depending on year/provider.
    if "EventName" in schedule.columns:
        event_names = schedule["EventName"].dropna().tolist()
    else:
        fallback_col = "OfficialEventName" if "OfficialEventName" in schedule.columns else schedule.columns[0]
        event_names = schedule[fallback_col].dropna().tolist()

    event_name = st.selectbox("Grand Prix", event_names)
    session_code = st.selectbox("Session", ["R", "Q", "S", "FP1", "FP2", "FP3"], index=0)

    st.divider()
    st.header("Chart settings")
    quicklaps_only = st.checkbox("Quicklaps only (Pace/Delta)", value=True)

# -----------------------------------------------------------------------------
# Coverage summary (informational)
# -----------------------------------------------------------------------------
st.subheader("📌 Data coverage")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Seasons available", f"{len(years)}")
col2.metric("From year", f"{min(years)}")
col3.metric("To year", f"{max(years)}")
col4.metric("Selected", f"{year} • {event_name} • {session_code}")
st.caption("Coverage reflects seasons for which an event schedule is accessible via the data provider.")
st.divider()

# -----------------------------------------------------------------------------
# Safe session loading (no raw tracebacks in UI)
# -----------------------------------------------------------------------------
with st.spinner("Loading session data..."):
    session, load_error = safe_load_session(year, event_name, session_code)

if load_error is not None:
    # Friendly message only (no exception text)
    st.warning(load_error)
    st.info("Please change the session or choose another event from the sidebar.")
    st.stop()

# From here onward, session exists and has lap data.
laps_all = session.laps
max_lap = int(laps_all["LapNumber"].max()) if "LapNumber" in laps_all.columns else 1

# -----------------------------------------------------------------------------
# Lap range selection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Lap filter")
    lap_range = st.slider("Lap range", 1, max_lap, (1, max_lap))

# -----------------------------------------------------------------------------
# Driver selection
# -----------------------------------------------------------------------------
driver_codes = sorted(list({session.get_driver(d)["Abbreviation"] for d in session.drivers}))

with st.sidebar:
    st.header("Drivers")
    d1 = st.selectbox("Driver 1", driver_codes, index=driver_codes.index("HAM") if "HAM" in driver_codes else 0)
    d2 = st.selectbox("Driver 2", driver_codes, index=driver_codes.index("VER") if "VER" in driver_codes else min(1, len(driver_codes) - 1))

name1 = safe_driver_fullname(session, d1)
name2 = safe_driver_fullname(session, d2)

# -----------------------------------------------------------------------------
# Build filtered lap datasets
#
# Two pipelines:
# - pace_laps_*: optional quicklaps filtering (cleaner charts)
# - strat_laps_*: no quicklaps filtering (correct stint detection)
# -----------------------------------------------------------------------------
pace_laps_1 = laps_for_pace(session.laps.pick_driver(d1), quicklaps_only, lap_range)
pace_laps_2 = laps_for_pace(session.laps.pick_driver(d2), quicklaps_only, lap_range)

strat_laps_1 = laps_for_strategy(session.laps.pick_driver(d1), lap_range)
strat_laps_2 = laps_for_strategy(session.laps.pick_driver(d2), lap_range)

# -----------------------------------------------------------------------------
# KPI row (based on pace-filtered laps)
# -----------------------------------------------------------------------------
kpi1 = compute_kpis(pace_laps_1)
kpi2 = compute_kpis(pace_laps_2)

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"{d1} Avg Lap", fmt_timedelta(kpi1["avg"]))
k2.metric(f"{d2} Avg Lap", fmt_timedelta(kpi2["avg"]))
k3.metric(f"{d1} Best Lap", fmt_timedelta(kpi1["best"]))
k4.metric(f"{d2} Best Lap", fmt_timedelta(kpi2["best"]))

st.caption(
    f"{d1}: {kpi1['laps']} laps in filter, {kpi1['pits']} pit-ins | "
    f"{d2}: {kpi2['laps']} laps in filter, {kpi2['pits']} pit-ins"
)
st.divider()

# =============================================================================
# Tabs
#
# Requested order:
# 1) Pace
# 2) Delta
# 3) Tyre Strategy
# 4) Raw Data
# =============================================================================
tab_pace, tab_delta, tab_strategy, tab_raw = st.tabs(
    ["Pace", "Delta", "Tyre Strategy", "Raw data"]
)

# -----------------------------------------------------------------------------
# TAB: Pace
# -----------------------------------------------------------------------------
with tab_pace:
    st.subheader(f"Pace: {name1} vs {name2}")

    fig = plt.figure()

    if not pace_laps_1.empty and pace_laps_1["LapTime"].notna().any():
        plt.plot(pace_laps_1["LapNumber"], td_series_to_seconds(pace_laps_1["LapTime"]), label=name1)

    if not pace_laps_2.empty and pace_laps_2["LapTime"].notna().any():
        plt.plot(pace_laps_2["LapNumber"], td_series_to_seconds(pace_laps_2["LapTime"]), label=name2)

    plt.xlabel("Lap")
    plt.ylabel("Lap Time (s)")
    plt.title(f"{event_name} {year} • {session_code} — Pace")
    plt.legend()
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# TAB: Delta
# -----------------------------------------------------------------------------
with tab_delta:
    st.subheader(f"Delta per lap: {name1} − {name2}")

    # Align both drivers by LapNumber so the delta is meaningful per lap
    merged = pd.merge(
        pace_laps_1[["LapNumber", "LapTime"]].dropna(),
        pace_laps_2[["LapNumber", "LapTime"]].dropna(),
        on="LapNumber",
        suffixes=("_1", "_2"),
    )

    if merged.empty:
        st.info("No overlapping laps found for delta computation with the current filters.")
    else:
        # Delta definition: Driver1 - Driver2
        merged["delta_s"] = (
            td_series_to_seconds(merged["LapTime_1"]) - td_series_to_seconds(merged["LapTime_2"])
        )

        fig2 = plt.figure()
        plt.axhline(0)
        plt.plot(merged["LapNumber"], merged["delta_s"])
        plt.xlabel("Lap")
        plt.ylabel(f"Seconds ({d1} − {d2})")
        plt.title(f"Delta ({name1} − {name2})  |  positive = {name1} slower")
        st.pyplot(fig2)

# -----------------------------------------------------------------------------
# TAB: Tyre Strategy
# -----------------------------------------------------------------------------
with tab_strategy:
    st.subheader("Tyre Strategy & Stints")
    st.caption(
        "Stints are detected from the full lap sequence (no quicklap filtering) "
        "to preserve compound changes around pit stops and restarts."
    )

    stints1 = build_stints(strat_laps_1)
    stints2 = build_stints(strat_laps_2)

    # Timeline visualization: each stint is a horizontal bar segment
    fig3 = plt.figure()
    y1, y2 = 1, 0
    bar_height = 0.35

    def draw_stints(stints: List[Dict[str, Any]], y: float) -> None:
        """
        Draw stints on a single row.

        - left: start lap
        - width: number of laps in stint
        - label: compound + stint length
        """
        for s in stints:
            start = int(s["start_lap"])
            length = int(s["length"])
            plt.barh(y, length, left=start, height=bar_height)
            mid = start + length / 2
            plt.text(mid, y, f"{s['compound']} ({length})", ha="center", va="center", fontsize=9)

    if stints1:
        draw_stints(stints1, y1)
    if stints2:
        draw_stints(stints2, y2)

    plt.yticks([y1, y2], [name1, name2])
    plt.xlabel("Lap")
    plt.title(f"{event_name} {year} • {session_code} — Tyre Strategy (selected lap range)")
    st.pyplot(fig3)

    st.divider()

    # Stint tables provide verification details for each detected stint
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{name1} ({d1}) — Stints**")
        if stints1:
            st.dataframe(pd.DataFrame(stints1), use_container_width=True)
        else:
            st.info("No tyre stint data found for this selection.")

    with c2:
        st.markdown(f"**{name2} ({d2}) — Stints**")
        if stints2:
            st.dataframe(pd.DataFrame(stints2), use_container_width=True)
        else:
            st.info("No tyre stint data found for this selection.")

# -----------------------------------------------------------------------------
# TAB: Raw data
# -----------------------------------------------------------------------------
with tab_raw:
    st.subheader("Filtered lap tables (pace filters)")

    # Raw tables are based on pace-filtered laps so they match the charts/KPIs.
    # This makes it easier for users to validate the plot inputs quickly.
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"**{name1} ({d1})**")
        cols_1 = [
            c for c in
            ["LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Compound", "TyreLife"]
            if c in pace_laps_1.columns
        ]
        st.dataframe(pace_laps_1[cols_1].copy(), use_container_width=True)

    with c2:
        st.markdown(f"**{name2} ({d2})**")
        cols_2 = [
            c for c in
            ["LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Compound", "TyreLife"]
            if c in pace_laps_2.columns
        ]
        st.dataframe(pace_laps_2[cols_2].copy(), use_container_width=True)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "If a session has no data (e.g., Sprint on a non-sprint weekend), "
    "the app displays a clean message instead of a traceback."
)
