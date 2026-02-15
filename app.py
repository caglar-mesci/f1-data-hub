import os
from typing import List, Dict, Any, Tuple

import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# F1 DATA HUB — Streamlit dashboard
#
# This file is intentionally written as a "single-file app":
# - Easier to run and deploy
#
# How Streamlit works:
# - Streamlit re-runs this entire script from top to bottom
#   every time you click something (buttons, selectboxes, sliders).
# - We use st.session_state to keep small pieces of state (like "which page").
# ============================================================


# ============================================================
# Page configuration
#
# layout="wide" is better for dashboards because it gives more horizontal space
# for charts and tables.
# ============================================================
st.set_page_config(page_title="F1 Data Hub", layout="wide")


# ============================================================
# FastF1 caching
#
# FastF1 downloads timing data from the data provider.
# Without a local cache:
# - First load is slow
# - Next loads are ALSO slow
#
# With a local cache:
# - First load downloads and saves files into ./cache
# - Next loads are much faster (files are reused)
# ============================================================
def ensure_fastf1_cache() -> None:
    """
    Create a local cache folder and enable FastF1 cache.
    This should be called before loading sessions.
    """
    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache("cache")


# ============================================================
# Time formatting helpers
#
# Pandas Timedelta normally prints like:
#   '0 days 00:01:23.456000'
#
# That looks ugly in a dashboard.
# We want a simple racing-style format:
#   '1:23.456' (mm:ss.mmm)
# ============================================================
def fmt_td(td: pd.Timedelta) -> str:
    """
    Convert a Timedelta to 'mm:ss.mmm' (or 'h:mm:ss.mmm' if needed).
    If td is missing, return 'N/A'.
    """
    if pd.isna(td):
        return "N/A"

    # Convert to total milliseconds (integer)
    total_ms = int(td.total_seconds() * 1000)

    # Keep sign for negative deltas (just in case)
    sign = "-" if total_ms < 0 else ""
    total_ms = abs(total_ms)

    ms = total_ms % 1000
    total_s = total_ms // 1000

    s = total_s % 60
    total_m = total_s // 60

    m = total_m % 60
    h = total_m // 60

    # Usually lap times don't have hours, but keep this for completeness
    if h > 0:
        return f"{sign}{h}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{sign}{m}:{s:02d}.{ms:03d}"


def td_to_seconds(td_series: pd.Series) -> pd.Series:
    """
    Matplotlib cannot plot Timedelta directly.
    Convert Timedelta -> float seconds.
    """
    return td_series.dt.total_seconds()


def safe_fullname(session, abbr: str) -> str:
    """
    Convert driver abbreviation (e.g., 'VER') into a full name.
    If metadata is missing, fall back to the abbreviation.
    """
    try:
        return session.get_driver(abbr)["FullName"]
    except Exception:
        return abbr


# ============================================================
# Cached data loaders
#
# st.cache_data:
# - Streamlit saves the returned value
# - If inputs do not change, it reuses the cached result
# - This avoids repeating expensive operations
# ============================================================
@st.cache_data(show_spinner=False)
def get_years_supported() -> List[int]:
    """
    Find seasons that have an accessible event schedule.

    We probe from 1950 to 2025 (inclusive range end is 2026 in code below).
    If the provider returns a schedule with events, we keep that year.

    This is cached so it does not re-scan every UI click.
    """
    years: List[int] = []

    for y in range(1950, 2026):
        try:
            sched = fastf1.get_event_schedule(y)
            if sched is not None and len(sched) > 0:
                years.append(y)
        except Exception:
            # Some years might not exist in the backend provider
            # We simply ignore them and continue scanning.
            pass

    return years


@st.cache_data(show_spinner=False)
def get_schedule(year: int) -> pd.DataFrame:
    """
    Return the full GP schedule for a given year.
    This is cached because schedules do not change inside a session.
    """
    return fastf1.get_event_schedule(year)


@st.cache_data(show_spinner=False)
def load_session(year: int, event_name: str, session_code: str):
    """
    Load a FastF1 session object and download required timing data.

    Inputs:
    - year: season year (e.g., 2021)
    - event_name: Grand Prix event name (e.g., 'Abu Dhabi')
    - session_code: session type ('R', 'Q', 'S', 'FP1', ...)

    This function is cached so if you re-open the same selection,
    it will not reload from scratch.
    """
    ensure_fastf1_cache()
    s = fastf1.get_session(year, event_name, session_code)
    s.load()
    return s


# ============================================================
# KPI / filtering helpers
# ============================================================
def compute_kpis(laps: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute simple statistics for a set of laps.

    Returns:
    - avg: average LapTime
    - best: minimum LapTime
    - pits: number of pit-ins (if PitInTime is available)
    - laps: number of laps in the current filtered view
    """
    lap_times = laps["LapTime"].dropna()
    avg = lap_times.mean() if not lap_times.empty else pd.NaT
    best = lap_times.min() if not lap_times.empty else pd.NaT

    pits = 0
    if "PitInTime" in laps.columns:
        pits = int(laps["PitInTime"].notna().sum())

    lap_count = int(laps["LapNumber"].nunique()) if "LapNumber" in laps.columns else 0

    return {"avg": avg, "best": best, "pits": pits, "laps": lap_count}


def filter_by_lap_range(laps: pd.DataFrame, lap_range: Tuple[int, int]) -> pd.DataFrame:
    """
    Keep only laps in the user-selected lap range.
    """
    if laps.empty:
        return laps
    if "LapNumber" not in laps.columns:
        return laps

    start, end = lap_range
    return laps[(laps["LapNumber"] >= start) & (laps["LapNumber"] <= end)]


def clean_laps_for_pace(laps: pd.DataFrame, quicklaps_only: bool, lap_range: Tuple[int, int]) -> pd.DataFrame:
    """
    Prepare laps for pace & delta charts.

    Why we might use quicklaps:
    - It removes laps that are typically not representative:
      * in-laps / out-laps
      * extreme slow laps
      * some outliers

    IMPORTANT:
    - quicklaps can remove laps around pit stops / restarts.
    - That is fine for "pace clarity",
      but it can break tyre strategy detection.
    """
    laps = filter_by_lap_range(laps, lap_range)

    if quicklaps_only:
        laps = laps.pick_quicklaps()

    return laps


def clean_laps_for_strategy(laps: pd.DataFrame, lap_range: Tuple[int, int]) -> pd.DataFrame:
    """
    Prepare laps for tyre strategy and stint detection.

    RULE:
    - Do NOT apply quicklaps here.
      We want the full lap sequence to detect compound changes correctly.

    Extra step:
    - Sometimes 'Compound' can be temporarily missing in a few rows.
      We forward-fill to avoid false stint breaks.
    """
    laps = filter_by_lap_range(laps, lap_range).sort_values("LapNumber").copy()

    if "Compound" in laps.columns:
        laps["Compound"] = laps["Compound"].ffill()

    return laps


# ============================================================
# Stint detection
# ============================================================
def build_stints(laps: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect tyre stints from lap data.

    A "stint" is a consecutive sequence of laps on the same compound.
    Example:
    - Medium for laps 1-14
    - Hard for laps 15-53
    - Soft for laps 54-58

    Output:
    A list of dictionaries, each describing one stint.
    """
    if laps.empty:
        return []

    if "Compound" not in laps.columns:
        return []

    df = laps.sort_values("LapNumber").copy()

    # If compound is missing everywhere, we cannot detect stints
    if df["Compound"].isna().all():
        return []

    stints: List[Dict[str, Any]] = []

    current_comp = None
    start_lap = None
    start_life = None

    last_lap = None
    last_life = None

    # Walk through laps in order and close stints when compound changes
    for _, row in df.iterrows():
        comp = row.get("Compound", None)

        # Skip rows where compound is still unknown
        if pd.isna(comp):
            continue

        lapn = int(row["LapNumber"])
        life = row.get("TyreLife", np.nan)

        # First valid lap initializes the first stint
        if current_comp is None:
            current_comp = comp
            start_lap = lapn
            start_life = life
            last_lap = lapn
            last_life = life
            continue

        # Compound change means the previous stint ended on the previous lap
        if comp != current_comp:
            stints.append(
                {
                    "compound": current_comp,
                    "start_lap": start_lap,
                    "end_lap": last_lap,
                    "length": last_lap - start_lap + 1,
                    "start_tyre_life": start_life,
                    "end_tyre_life": last_life,
                }
            )

            # Start a new stint from this lap
            current_comp = comp
            start_lap = lapn
            start_life = life

        # Always update "last seen" lap info
        last_lap = lapn
        last_life = life

    # Close the final stint after the loop
    if current_comp is not None and start_lap is not None and last_lap is not None:
        stints.append(
            {
                "compound": current_comp,
                "start_lap": start_lap,
                "end_lap": last_lap,
                "length": last_lap - start_lap + 1,
                "start_tyre_life": start_life,
                "end_tyre_life": last_life,
            }
        )

    return stints


# ============================================================
# Simple "page" routing using session_state
#
# We keep 2 pages:
# - homepage: introduction and quick start
# - explore: the dashboard itself
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "homepage"


def goto(page_name: str) -> None:
    """
    Change the current page.
    We call st.rerun() where buttons trigger navigation
    to ensure it works on the first click.
    """
    st.session_state.page = page_name


# ============================================================
# HOMEPAGE
# ============================================================
if st.session_state.page == "homepage":
    st.title("🏁 F1 Data Hub")

    # Short, intro text in English
    st.markdown(
        """
### Formula 1 Data Exploration & Comparison Platform

Use this dashboard to:
- Select **Season → Grand Prix → Session**
- Compare **two drivers** with pace and delta charts
- Inspect **tyre strategy** with stint detection

Data source: FastF1 timing data.
"""
    )

    # A small, curated "history" section for the homepage
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

    # Navigation buttons
    c1, c2 = st.columns([1, 2])
    with c1:
        # st.rerun() guarantees immediate navigation (no "second click" issue)
        if st.button("Enter Platform →", use_container_width=True):
            goto("explore")
            st.rerun()
    with c2:
        st.info("First data load may take a few seconds. Later runs are faster due to caching.")

    st.divider()

    # Signature / credits section
    st.markdown(
        """
**Built by Çağlar Mesci**  
GitHub: https://github.com/caglar-mesci
"""
    )

    st.stop()


# ============================================================
# EXPLORE PAGE (Main dashboard)
# ============================================================
st.title("🏎️ Explore & Compare")

# Load supported seasons once (cached)
years = get_years_supported()
if not years:
    st.error("No supported seasons found from the data provider.")
    st.stop()

# Sidebar holds all interactive controls
with st.sidebar:
    st.header("Navigation")

    # Go back to homepage
    if st.button("🏠 Homepage", use_container_width=True):
        goto("homepage")
        st.rerun()

    st.divider()
    st.header("Race selection")

    # Default year choice for convenience
    default_year = 2021 if 2021 in years else max(years)
    year = st.selectbox("Season", years, index=years.index(default_year))

    # Load schedule for selected year (cached)
    sched = get_schedule(year)

    # Different years/providers may use different column names
    if "EventName" in sched.columns:
        event_names = sched["EventName"].dropna().tolist()
    else:
        fallback_col = "OfficialEventName" if "OfficialEventName" in sched.columns else sched.columns[0]
        event_names = sched[fallback_col].dropna().tolist()

    event_name = st.selectbox("Grand Prix", event_names)
    session_code = st.selectbox("Session", ["R", "Q", "S", "FP1", "FP2", "FP3"], index=0)

    st.divider()
    st.header("Chart settings")

    # This toggle affects ONLY pace/delta charts, not tyre strategy.
    quicklaps_only = st.checkbox("Quicklaps only (Pace/Delta)", value=True)

# Small coverage summary (based on which schedules are accessible)
st.subheader("📌 Data coverage")
colA, colB, colC, colD = st.columns(4)
colA.metric("Seasons available", f"{len(years)}")
colB.metric("From year", f"{min(years)}")
colC.metric("To year", f"{max(years)}")
colD.metric("Selected", f"{year} • {event_name} • {session_code}")
st.caption("Coverage reflects seasons for which an event schedule is accessible via the data provider.")
st.divider()

# Load session timing data (can take time on first run)
with st.spinner("Loading session data..."):
    session = load_session(year, event_name, session_code)

# Grab laps table
laps_all = session.laps
if laps_all.empty:
    st.error("No lap data found for the selected session.")
    st.stop()

# Lap slider needs a max lap number
max_lap = int(laps_all["LapNumber"].max()) if "LapNumber" in laps_all.columns else 1

with st.sidebar:
    st.header("Lap filter")
    lap_range = st.slider("Lap range", 1, max_lap, (1, max_lap))

# Build driver dropdowns from session participants
driver_codes = sorted(list({session.get_driver(d)["Abbreviation"] for d in session.drivers}))

with st.sidebar:
    st.header("Drivers")

    # Choose two drivers for comparison
    d1 = st.selectbox("Driver 1", driver_codes, index=driver_codes.index("HAM") if "HAM" in driver_codes else 0)
    d2 = st.selectbox("Driver 2", driver_codes, index=driver_codes.index("VER") if "VER" in driver_codes else min(1, len(driver_codes) - 1))

# Resolve abbreviations to full names for labels
name1 = safe_fullname(session, d1)
name2 = safe_fullname(session, d2)

# Create TWO versions of lap data:
# - pace_laps_* for pace/delta charts (can use quicklaps)
# - strat_laps_* for tyre strategy (must NOT use quicklaps)
pace_laps_1 = clean_laps_for_pace(session.laps.pick_driver(d1), quicklaps_only, lap_range)
pace_laps_2 = clean_laps_for_pace(session.laps.pick_driver(d2), quicklaps_only, lap_range)

strat_laps_1 = clean_laps_for_strategy(session.laps.pick_driver(d1), lap_range)
strat_laps_2 = clean_laps_for_strategy(session.laps.pick_driver(d2), lap_range)

# KPIs are usually better aligned with the pace chart filters
kpi1 = compute_kpis(pace_laps_1)
kpi2 = compute_kpis(pace_laps_2)

# Show KPIs at the top
k1, k2, k3, k4 = st.columns(4)
k1.metric(f"{d1} Avg Lap", fmt_td(kpi1["avg"]))
k2.metric(f"{d2} Avg Lap", fmt_td(kpi2["avg"]))
k3.metric(f"{d1} Best Lap", fmt_td(kpi1["best"]))
k4.metric(f"{d2} Best Lap", fmt_td(kpi2["best"]))

st.caption(
    f"{d1}: {kpi1['laps']} laps in filter, {kpi1['pits']} pit-ins | "
    f"{d2}: {kpi2['laps']} laps in filter, {kpi2['pits']} pit-ins"
)
st.divider()

# Tabs separate analysis views
tab1, tab2, tab3, tab4 = st.tabs(["Pace", "Delta", "Raw data", "Tyre Strategy"])


# ============================================================
# TAB 1: PACE
#
# We plot lap number vs lap time (in seconds) for both drivers.
# ============================================================
with tab1:
    st.subheader(f"Pace: {name1} vs {name2}")

    fig = plt.figure()

    # Plot driver 1 pace line
    if not pace_laps_1.empty and pace_laps_1["LapTime"].notna().any():
        plt.plot(pace_laps_1["LapNumber"], td_to_seconds(pace_laps_1["LapTime"]), label=name1)

    # Plot driver 2 pace line
    if not pace_laps_2.empty and pace_laps_2["LapTime"].notna().any():
        plt.plot(pace_laps_2["LapNumber"], td_to_seconds(pace_laps_2["LapTime"]), label=name2)

    plt.xlabel("Lap")
    plt.ylabel("Lap Time (s)")
    plt.title(f"{event_name} {year} • {session_code} — Pace")
    plt.legend()
    st.pyplot(fig)


# ============================================================
# TAB 2: DELTA
#
# Delta is computed only on laps where BOTH drivers have a valid LapTime.
# delta_s = driver1_time - driver2_time
#   positive => driver1 slower
#   negative => driver1 faster
# ============================================================
with tab2:
    st.subheader(f"Delta per lap: {name1} − {name2}")

    # Merge on lap number to align laps
    merged = pd.merge(
        pace_laps_1[["LapNumber", "LapTime"]].dropna(),
        pace_laps_2[["LapNumber", "LapTime"]].dropna(),
        on="LapNumber",
        suffixes=("_1", "_2"),
    )

    if merged.empty:
        st.info("No overlapping laps found for delta computation with the current filters.")
    else:
        merged["delta_s"] = td_to_seconds(merged["LapTime_1"]) - td_to_seconds(merged["LapTime_2"])

        fig2 = plt.figure()
        plt.axhline(0)
        plt.plot(merged["LapNumber"], merged["delta_s"])
        plt.xlabel("Lap")
        plt.ylabel(f"Seconds ({d1} − {d2})")
        plt.title(f"Delta ({name1} − {name2})  |  positive = {name1} slower")
        st.pyplot(fig2)


# ============================================================
# TAB 3: RAW DATA
#
# Show a compact, readable table so users can verify the numbers.
# We show common columns when available.
# ============================================================
with tab3:
    st.subheader("Filtered lap tables (pace filters)")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"**{name1} ({d1})**")
        cols = [c for c in ["LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Compound", "TyreLife"] if c in pace_laps_1.columns]
        st.dataframe(pace_laps_1[cols].copy(), use_container_width=True)

    with c2:
        st.markdown(f"**{name2} ({d2})**")
        cols = [c for c in ["LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Compound", "TyreLife"] if c in pace_laps_2.columns]
        st.dataframe(pace_laps_2[cols].copy(), use_container_width=True)


# ============================================================
# TAB 4: TYRE STRATEGY
#
# Stint detection MUST use the full lap sequence (no quicklaps),
# otherwise compound changes near pit/restart laps can disappear.
#
# This tab uses strat_laps_* data (not pace_laps_*).
# ============================================================
with tab4:
    st.subheader("Tyre Strategy & Stints")
    st.caption(
        "Stints are detected as consecutive laps on the same compound within the selected lap range. "
        "This view ignores quicklap filtering to preserve compound changes."
    )

    # Build stints from strategy-safe laps
    stints1 = build_stints(strat_laps_1)
    stints2 = build_stints(strat_laps_2)

    # --- Timeline chart ---
    # We draw horizontal bars for each stint:
    # - Bar length = number of laps in the stint
    # - Bar position (left) = start lap
    # - Text inside bar = compound and stint length
    fig3 = plt.figure()

    # Two rows: one per driver
    y_driver1, y_driver2 = 1, 0
    bar_height = 0.35

    def draw_stints(stints: List[Dict[str, Any]], y: float) -> None:
        """
        Draw stints as horizontal bars on a single row (y coordinate).
        Each bar corresponds to one stint.
        """
        for s in stints:
            start = int(s["start_lap"])
            length = int(s["length"])

            # Draw bar segment
            plt.barh(y, length, left=start, height=bar_height)

            # Label bar with compound and lap count
            mid = start + length / 2
            plt.text(mid, y, f"{s['compound']} ({length})", ha="center", va="center", fontsize=9)

    # Draw both drivers if data exists
    if stints1:
        draw_stints(stints1, y_driver1)
    if stints2:
        draw_stints(stints2, y_driver2)

    # Label y-axis with full names
    plt.yticks([y_driver1, y_driver2], [name1, name2])
    plt.xlabel("Lap")
    plt.title(f"{event_name} {year} • {session_code} — Tyre Strategy (selected lap range)")
    st.pyplot(fig3)

    st.divider()

    # --- Stint tables ---
    # Tables help users verify:
    # - where each stint starts/ends
    # - how many laps the stint lasts
    # - tyre life values at start/end when available
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"**{name1} ({d1}) — Stints**")
        if stints1:
            st.dataframe(pd.DataFrame(stints1), use_container_width=True)
        else:
            st.info("No stint data available (compound may be missing for this session/filter).")

    with c2:
        st.markdown(f"**{name2} ({d2}) — Stints**")
        if stints2:
            st.dataframe(pd.DataFrame(stints2), use_container_width=True)
        else:
            st.info("No stint data available (compound may be missing for this session/filter).")


# ============================================================
# Footer note (kept short)
# ============================================================
st.divider()
st.caption("Planned enhancements: sector comparison, safety car split, and CSV export for filtered laps.")
