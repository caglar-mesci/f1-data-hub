import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

def fmt_timedelta(td: pd.Timedelta) -> str:
    """
    Convert a pandas Timedelta to a racing-style string: mm:ss.mmm
    If input is NaT or empty, it returns 'N/A'
    """
    if pd.isna(td):
        return "N/A"
    try:
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
    except Exception:
        return str(td)

def format_dataframe_timedeltas(df: pd.DataFrame) -> pd.DataFrame:
    """Formats all pandas Timedelta columns to clean text mm:ss.mmm format"""
    new_df = df.copy()
    for col in new_df.columns:
        if pd.api.types.is_timedelta64_dtype(new_df[col]):
            new_df[col] = new_df[col].apply(fmt_timedelta)
    return new_df


def td_series_to_seconds(td_series: pd.Series) -> pd.Series:
    """Matplotlib cannot directly plot Timedelta values."""
    return td_series.dt.total_seconds()


def safe_driver_fullname(session: Any, abbr: str) -> str:
    """Map a driver abbreviation (e.g., 'VER') to their full driver name."""
    try:
        return session.get_driver(abbr)["FullName"]
    except Exception:
        return abbr


def get_fastest_sectors(laps: pd.DataFrame) -> Dict[str, str]:
    """Returns fastest sector times formatted clearly, for text metrics."""
    res = {"s1": "N/A", "s2": "N/A", "s3": "N/A"}
    for sect, k in [("Sector1Time", "s1"), ("Sector2Time", "s2"), ("Sector3Time", "s3")]:
        if sect in laps.columns:
            s_times = laps[sect].dropna()
            if not s_times.empty:
                res[k] = fmt_timedelta(s_times.min())
    return res

def filter_by_lap_range(laps: pd.DataFrame, lap_range: Tuple[int, int]) -> pd.DataFrame:
    """Filter laps by inclusive LapNumber range."""
    if laps.empty or "LapNumber" not in laps.columns:
        return laps
    start, end = lap_range
    return laps[(laps["LapNumber"] >= start) & (laps["LapNumber"] <= end)]


def laps_for_pace(laps: pd.DataFrame, quicklaps_only: bool, lap_range: Tuple[int, int]) -> pd.DataFrame:
    """Prepare laps for pace and delta charts."""
    laps = filter_by_lap_range(laps, lap_range)
    if quicklaps_only:
        laps = laps.pick_quicklaps()
    return laps


def laps_for_strategy(laps: pd.DataFrame, lap_range: Tuple[int, int]) -> pd.DataFrame:
    """Prepare laps for tyre strategy analysis."""
    laps = filter_by_lap_range(laps, lap_range).sort_values("LapNumber").copy()
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


def build_stints(laps: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect tyre stints from lap data."""
    if laps.empty or "Compound" not in laps.columns:
        return []

    df = laps.sort_values("LapNumber").copy()
    if df["Compound"].isna().all():
        return []

    # Vectorized approach to stint detection is better, but a simple 
    # forward iteration is fine for 50-70 laps and ensures edge-cases are caught
    # We will stick to iterrows here as it's safe and small performance hit per driver.
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

        if current_comp is None:
            current_comp = comp
            start_lap = lapn
            start_life = life
            last_lap = lapn
            last_life = life
            continue

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

        last_lap = lapn
        last_life = life

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
