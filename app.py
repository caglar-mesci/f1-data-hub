import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import get_supported_years, get_year_schedule, safe_load_session, get_available_sessions
from utils import (
    fmt_timedelta, td_series_to_seconds, safe_driver_fullname, 
    laps_for_pace, laps_for_strategy, compute_kpis, build_stints, 
    format_dataframe_timedeltas, get_fastest_sectors
)

def setup_page_config() -> None:
    """Configures the initial Streamlit page settings and injects custom CSS."""
    st.set_page_config(
        page_title="F1 Data Hub", 
        page_icon="🏁", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
        <style>
        .metric-container {
            border-radius: 10px;
            padding: 5px;
            background-color: rgba(255, 255, 255, 0.05);
        }
        .stTabs [role="tablist"] {
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }
        .stTabs [role="tab"] {
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

def goto(page_name: str) -> None:
    """Updates the session state to navigate between pages."""
    st.session_state.page = page_name

def render_homepage() -> None:
    """Renders the aesthetic landing page with feature cards and champions."""
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0;">🏁 <span style="color: #e10600;">F1</span> Data Hub</h1>
        <p style="font-size: 1.1rem; color: #a3a8b8; font-weight: 300; margin-top: 5px;">Formula 1 Data Exploration & Comparison Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3)
    f1.info("🏎️ **Deep Comparison**\n\nCompare two drivers dynamically using pace and delta charts.")
    f2.info("🔄 **Strategy Insights**\n\nReview tyre strategy with automatic tyre stint detection.")
    f3.info("📊 **Reliable Data**\n\nAll data is powered and verified by official **FastF1** timing datasets.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 🏆 Most World Championships")
    st.markdown("""
    <style>
    .champ-card {
        padding: 10px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .champ-card:hover {
        transform: translateY(-4px);
        border-color: #e10600;
        box-shadow: 0 4px 10px rgba(225, 6, 0, 0.2);
    }
    .champ-num {
        font-size: 1.3rem;
        font-weight: 900;
        color: #e10600;
    }
    .champ-name {
        font-size: 1rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='champ-card'><span class='champ-num'>7×</span><br><span class='champ-name'>Michael Schumacher</span></div>", unsafe_allow_html=True)
    c2.markdown("<div class='champ-card'><span class='champ-num'>7×</span><br><span class='champ-name'>Lewis Hamilton</span></div>", unsafe_allow_html=True)
    c3.markdown("<div class='champ-card'><span class='champ-num'>5×</span><br><span class='champ-name'>Juan Manuel Fangio</span></div>", unsafe_allow_html=True)
    
    c4, c5, c6 = st.columns(3)
    c4.markdown("<div class='champ-card'><span class='champ-num'>4×</span><br><span class='champ-name'>Alain Prost</span></div>", unsafe_allow_html=True)
    c5.markdown("<div class='champ-card'><span class='champ-num'>4×</span><br><span class='champ-name'>Sebastian Vettel</span></div>", unsafe_allow_html=True)
    c6.markdown("<div class='champ-card'><span class='champ-num'>4×</span><br><span class='champ-name'>Max Verstappen</span></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🚀 Enter Dashboard", use_container_width=True, type="primary"):
            goto("explore")
            st.rerun()
    with col2:
        st.markdown(
            "<div style='padding-top: 7px; color: #a3a8b8; font-size: 0.95em;'>"
            "ℹ️ <i><b>Note:</b> The initial data fetching for a session might take some time to download and cache.</i>"
            "</div>", 
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #a3a8b8; padding: 10px;'>"
        "<b>Built by Çağlar Mesci</b> &nbsp; | &nbsp; "
        "<a href='https://github.com/caglar-mesci' style='color: white; text-decoration: none;'>🔗 GitHub Profile</a>"
        "</div>", 
        unsafe_allow_html=True
    )

def render_dashboard(year: int, event_name: str, session_name: str, quicklaps_only: bool) -> None:
    """Fetches telemetry data and renders the driver comparison dashboard."""
    st.title("🏎️ Driver Comparison Dashboard")

    # Fetch Data Smoothly
    with st.status("Fetching race telemetry...", expanded=True) as status:
        st.write(f"Connecting to FastF1 cache for {year} {event_name} - {session_name}...")
        session, load_error = safe_load_session(year, event_name, session_name)
        
        if load_error:
            status.update(label="There is no data available for this season/event.", state="error", expanded=False)
            st.stop()
        else:
            st.write("Telemetry loaded successfully.")
            status.update(label="Download Complete", state="complete", expanded=False)

    laps_all = session.laps
    max_lap = int(laps_all["LapNumber"].max()) if (not laps_all.empty and "LapNumber" in laps_all.columns) else 1

    # Dependent Sidebar Filters (Drivers and Laps)
    with st.sidebar:
        st.header("Filters")
        lap_range = st.slider("Lap range", 1, max_lap, (1, max_lap))
        
        driver_codes = sorted(list({session.get_driver(d)["Abbreviation"] for d in session.drivers}))
        if not driver_codes:
            st.error("No drivers found in this session.")
            st.stop()
            
        d1 = st.selectbox("Driver 1", driver_codes, index=driver_codes.index("HAM") if "HAM" in driver_codes else 0)
        d2 = st.selectbox("Driver 2", driver_codes, index=driver_codes.index("VER") if "VER" in driver_codes else min(1, len(driver_codes) - 1))

    # Calculate Laps
    name1 = safe_driver_fullname(session, d1)
    name2 = safe_driver_fullname(session, d2)

    pace_laps_1 = laps_for_pace(session.laps.pick_driver(d1), quicklaps_only, lap_range)
    pace_laps_2 = laps_for_pace(session.laps.pick_driver(d2), quicklaps_only, lap_range)

    strat_laps_1 = laps_for_strategy(session.laps.pick_driver(d1), lap_range)
    strat_laps_2 = laps_for_strategy(session.laps.pick_driver(d2), lap_range)

    # -----------------------------------------------------------------------------
    # Render Comparison Panel
    # -----------------------------------------------------------------------------
    st.subheader(f"📊 Driver Duel: {name1} vs {name2}")

    kpi1 = compute_kpis(pace_laps_1)
    kpi2 = compute_kpis(pace_laps_2)
    fast_sec_1 = get_fastest_sectors(pace_laps_1)
    fast_sec_2 = get_fastest_sectors(pace_laps_2)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 🏎️ {name1} ({d1})")
        m1, m2, m3 = st.columns(3)
        m1.metric("Best Lap", fmt_timedelta(kpi1["best"]))
        m2.metric("Avg Pace", fmt_timedelta(kpi1["avg"]))
        m3.metric("Laps Analyzed", kpi1["laps"])
        
        st.markdown("**Fastest Sectors:**")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Sector 1", fast_sec_1["s1"])
        sc2.metric("Sector 2", fast_sec_1["s2"])
        sc3.metric("Sector 3", fast_sec_1["s3"])

    with col2:
        st.markdown(f"### 🏎️ {name2} ({d2})")
        m1, m2, m3 = st.columns(3)
        m1.metric("Best Lap", fmt_timedelta(kpi2["best"]))
        m2.metric("Avg Pace", fmt_timedelta(kpi2["avg"]))
        m3.metric("Laps Analyzed", kpi2["laps"])
        
        st.markdown("**Fastest Sectors:**")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Sector 1", fast_sec_2["s1"])
        sc2.metric("Sector 2", fast_sec_2["s2"])
        sc3.metric("Sector 3", fast_sec_2["s3"])

    st.divider()

    # -----------------------------------------------------------------------------
    # Detail Analysis Tabs
    # -----------------------------------------------------------------------------
    tab_pace, tab_sector, tab_delta, tab_strategy, tab_raw = st.tabs(
        ["Pace", "Sector Analysis", "Delta", "Tyre Strategy", "Clean Data"]
    )

    with tab_pace:
        st.markdown("#### Track Pace History")
        fig1 = plt.figure(figsize=(10,4))
        if not pace_laps_1.empty and pace_laps_1["LapTime"].notna().any():
            plt.plot(pace_laps_1["LapNumber"], td_series_to_seconds(pace_laps_1["LapTime"]), label=name1, color='blue', alpha=0.8)
        if not pace_laps_2.empty and pace_laps_2["LapTime"].notna().any():
            plt.plot(pace_laps_2["LapNumber"], td_series_to_seconds(pace_laps_2["LapTime"]), label=name2, color='red', alpha=0.8)
        plt.xlabel("Lap")
        plt.ylabel("Seconds")
        plt.legend()
        plt.grid(True, alpha=0.2)
        st.pyplot(fig1)

    with tab_sector:
        st.markdown("#### Sector-by-Sector Comparisons")
        fig_sec, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
        
        def plot_sector(axis, data1, data2, col_name, title):
            """Helper function to plot individual sectors."""
            if col_name in data1.columns and not data1.empty:
                 axis.plot(data1["LapNumber"], td_series_to_seconds(data1[col_name]), color='blue', alpha=0.7, label=d1)
            if col_name in data2.columns and not data2.empty:
                 axis.plot(data2["LapNumber"], td_series_to_seconds(data2[col_name]), color='red', alpha=0.7, label=d2)
            axis.set_title(title)
            axis.grid(True, alpha=0.2)
            
        plot_sector(ax1, pace_laps_1, pace_laps_2, "Sector1Time", "Sector 1")
        plot_sector(ax2, pace_laps_1, pace_laps_2, "Sector2Time", "Sector 2")
        plot_sector(ax3, pace_laps_1, pace_laps_2, "Sector3Time", "Sector 3")
        
        ax1.set_ylabel("Seconds")
        ax2.legend()
        st.pyplot(fig_sec)

    with tab_delta:
        st.markdown("#### Lap Delta Time")
        merged = pd.merge(
            pace_laps_1[["LapNumber", "LapTime"]].dropna(),
            pace_laps_2[["LapNumber", "LapTime"]].dropna(),
            on="LapNumber", suffixes=("_1", "_2"),
        )
        if merged.empty:
            st.info("No matching laps found for delta visualization.")
        else:
            merged["delta_s"] = td_series_to_seconds(merged["LapTime_1"]) - td_series_to_seconds(merged["LapTime_2"])
            fig2 = plt.figure(figsize=(10,4))
            plt.axhline(0, color='gray', linestyle='--')
            plt.fill_between(merged["LapNumber"], merged["delta_s"], 0, where=(merged["delta_s"] > 0), color='red', alpha=0.5, label=f'{name1} Slower')
            plt.fill_between(merged["LapNumber"], merged["delta_s"], 0, where=(merged["delta_s"] <= 0), color='blue', alpha=0.5, label=f'{name2} Slower')
            plt.plot(merged["LapNumber"], merged["delta_s"], color="black", linewidth=1)
            plt.xlabel("Lap")
            plt.ylabel(f"Gap (Seconds) + = {d1} Slower")
            plt.legend()
            st.pyplot(fig2)

    with tab_strategy:
        st.markdown("#### Tyre Degradation & Strategy")
        stints1 = build_stints(strat_laps_1)
        stints2 = build_stints(strat_laps_2)

        fig3 = plt.figure(figsize=(10,2))
        def draw_stints(stints, y):
            """Helper function to draw tyre stints onto a timeline."""
            colors = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "whitesmoke", "INTERMEDIATE": "green", "WET": "blue"}
            for s in stints:
                c = str(s["compound"]).upper()
                bar_color = colors.get(c, "grey")
                plt.barh(y, s["length"], left=s["start_lap"], height=0.4, color=bar_color, edgecolor="black")
                plt.text(s["start_lap"] + s["length"]/2, y, f"{c[0]} ({s['length']})", ha="center", va="center", color="black", fontsize=8, fontweight="bold")

        if stints1: draw_stints(stints1, 1)
        if stints2: draw_stints(stints2, 0)
        plt.yticks([1, 0], [name1, name2])
        plt.xlabel("Lap")
        # Setting dark background for tyre stints to look good even for white 'HARD' tyres
        plt.gca().set_facecolor('#222')
        st.pyplot(fig3)
        
        cl1, cl2 = st.columns(2)
        with cl1:
            if stints1:
                st.dataframe(pd.DataFrame(stints1), hide_index=True)
            else:
                st.write("No stints")
        with cl2:
            if stints2:
                st.dataframe(pd.DataFrame(stints2), hide_index=True)
            else:
                st.write("No stints")

    with tab_raw:
        st.markdown("#### Cleaned Lap Times (mm:ss.mmm)")
        c1, c2 = st.columns(2)
        display_cols = ["LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Compound", "TyreLife"]
        
        with c1:
            st.markdown(f"**{name1}**")
            df1 = pace_laps_1[[c for c in display_cols if c in pace_laps_1.columns]].copy()
            st.dataframe(format_dataframe_timedeltas(df1), use_container_width=True, hide_index=True)
        with c2:
            st.markdown(f"**{name2}**")
            df2 = pace_laps_2[[c for c in display_cols if c in display_cols if c in pace_laps_2.columns]].copy()
            st.dataframe(format_dataframe_timedeltas(df2), use_container_width=True, hide_index=True)

def main() -> None:
    """Main execution flow for the Streamlit application."""
    setup_page_config()

    if "page" not in st.session_state:
        st.session_state.page = "homepage"

    if st.session_state.page == "homepage":
        render_homepage()
        return

    # Handle global sidebar settings when in explore mode
    years = get_supported_years()
    if not years:
        st.error("No data.")
        st.stop()

    with st.sidebar:
        if st.button("🏠 Home", use_container_width=True):
            goto("homepage")
            st.rerun()

        st.header("⚙️ Settings")
        
        # Determine the most recent year cleanly
        default_year = 2021 if 2021 in years else years[0]
        selected_year = st.selectbox("Season", years, index=years.index(default_year))

        schedule = get_year_schedule(selected_year)
        if "EventName" in schedule.columns:
            event_names = schedule["EventName"].dropna().tolist()
        else:
            fallback_col = "OfficialEventName" if "OfficialEventName" in schedule.columns else schedule.columns[0]
            event_names = schedule[fallback_col].dropna().tolist()

        selected_event = st.selectbox("Grand Prix", event_names)
        
        available_sessions = get_available_sessions(selected_year, selected_event)
        
        if not available_sessions:
             st.error("No data.")
             st.stop()
        else:
             selected_session = st.selectbox("Session", available_sessions)
        
        st.divider()
        quicklaps_only = st.checkbox("Quicklaps only (Pace filters)", value=True)

    # Proceed to rendering the specific dashboard logic using the selected parameters
    render_dashboard(selected_year, selected_event, selected_session, quicklaps_only)

if __name__ == "__main__":
    main()
