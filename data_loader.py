import os
from typing import List, Tuple, Optional, Any
import fastf1
import pandas as pd
import streamlit as st

"""
data_loader.py
---------------
This module manages all connections to the FastF1 API. It hides complex try/except blocks,
enables local caching, and handles parsing available sessions before sending 
them back to the main UI.

This separation of concerns ensures app.py remains free of pure data-fetching logic.
"""

def ensure_fastf1_cache() -> None:
    """Create and enable a local cache folder for FastF1."""
    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache("cache")

@st.cache_data(show_spinner=False)
def get_supported_years() -> List[int]:
    """Discover seasons that have a non-empty event schedule."""
    years: List[int] = []
    # Test a reasonable range
    for y in range(1950, 2026):
        try:
            sched = fastf1.get_event_schedule(y)
            if sched is not None and len(sched) > 0:
                years.append(y)
        except Exception:
            pass
    return sorted(years, reverse=True) # Let's sort them descending for better UX

@st.cache_data(show_spinner=False)
def get_year_schedule(year: int) -> pd.DataFrame:
    """Return the event schedule for a specific season year (cached)."""
    return fastf1.get_event_schedule(year)

@st.cache_data(show_spinner=False)
def get_available_sessions(year: int, event_name: str) -> List[str]:
    """
    Given a year and event name, fetch the event and see which sessions
    actually exist. FastF1 events have `Session1`, `Session2` etc.
    """
    try:
        ensure_fastf1_cache()
        event = fastf1.get_event(year, event_name)
        sessions = []
        for i in range(1, 6):
            s_name = getattr(event, f'Session{i}', None)
            # nan checking for pandas
            if s_name and not pd.isna(s_name) and str(s_name) != 'nan':
                sessions.append(str(s_name))
        return sessions
    except Exception:
        # Fallback to standard weekend if there's an api failure
        return ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]

@st.cache_data(show_spinner=False)
def safe_load_session(year: int, event_name: str, session_name: str) -> Tuple[Any, Optional[str]]:
    """
    Load a FastF1 session safely and elegantly handle API errors.
    Returns: (session_object, error_message)
    """
    try:
        ensure_fastf1_cache()
        s = fastf1.get_session(year, event_name, session_name)
        s.load()

        if s.laps is None or s.laps.empty:
            return None, "There is no data available for this season/event."
        return s, None
    except Exception as e:
        return None, "There is no data available for this season/event."
