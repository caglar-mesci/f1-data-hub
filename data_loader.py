import os
from typing import List, Tuple, Optional, Any, Dict
from datetime import datetime
import json
import urllib.request
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
    """Return supported Formula 1 seasons from current year down to 1950."""
    current_year = datetime.now().year
    return list(range(current_year, 1949, -1))

def _http_get_json(url: str, timeout: int = 3) -> Optional[dict]:
    """Helper to safely fetch JSON from external endpoints using standard urllib."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "F1DataHub/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                return json.loads(response.read().decode("utf-8"))
    except Exception:
        pass
    return None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_live_champion_stats() -> Dict[str, Dict[str, Any]]:
    """
    Fetches up-to-date driver career statistics (Races, Wins, Poles, Podiums)
    from the Ergast / Jolpica API. Cached for 24 hours (86400 seconds).
    Returns a dictionary mapping driver display names to updated stat fields.
    """
    driver_map = {
        "Lewis Hamilton": "hamilton",
        "Max Verstappen": "max_verstappen",
        "Sebastian Vettel": "vettel",
        "Michael Schumacher": "schumacher",
        "Alain Prost": "prost",
        "Juan Manuel Fangio": "fangio"
    }
    live_stats: Dict[str, Dict[str, Any]] = {}
    for name, driver_id in driver_map.items():
        try:
            d_data: Dict[str, Any] = {}
            # Total races
            races_json = _http_get_json(f"https://api.jolpi.ca/ergast/f1/drivers/{driver_id}/results.json?limit=1")
            if races_json and "MRData" in races_json:
                d_data["races"] = int(races_json["MRData"]["total"])

            # Wins & Podiums (p1, p2, p3)
            p1_json = _http_get_json(f"https://api.jolpi.ca/ergast/f1/drivers/{driver_id}/results/1.json?limit=1")
            p2_json = _http_get_json(f"https://api.jolpi.ca/ergast/f1/drivers/{driver_id}/results/2.json?limit=1")
            p3_json = _http_get_json(f"https://api.jolpi.ca/ergast/f1/drivers/{driver_id}/results/3.json?limit=1")

            if p1_json and p2_json and p3_json:
                p1 = int(p1_json["MRData"]["total"])
                p2 = int(p2_json["MRData"]["total"])
                p3 = int(p3_json["MRData"]["total"])
                d_data["wins"] = p1
                d_data["podiums"] = p1 + p2 + p3

            # Poles (qualifying P1)
            poles_json = _http_get_json(f"https://api.jolpi.ca/ergast/f1/drivers/{driver_id}/qualifying/1.json?limit=1")
            if poles_json and "MRData" in poles_json:
                poles_cnt = int(poles_json["MRData"]["total"])
                if poles_cnt > 0:
                    d_data["poles"] = poles_cnt

            if d_data:
                live_stats[name] = d_data
        except Exception:
            pass
    return live_stats



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
