# 🏁 F1 Data Hub

[![Live Demo](https://img.shields.io/badge/Live-Demo-red?style=for-the-badge)](https://f1-data-caglarmesci.streamlit.app/)

An interactive Formula 1 race analytics dashboard built with **FastF1** and **Streamlit**. 

This project transforms official timing data into a structured analytical interface for performance benchmarking, sector analysis, tyre strategy evaluation, and AI-driven session summaries.

---

## 🌐 Live Demo
Play with the live app here: https://f1-data-caglarmesci.streamlit.app/

---

## 🚀 Key Features

*   **Dynamic Session Loading:** Intelligent filtering that strictly retrieves actual past sessions (e.g., skips non-existent "Sprint" sessions dynamically).
*   **Advanced Driver Duel Dashboard:** In-depth metrics comparing two drivers side-by-side (Best Lap, Average Pace, Fastest Sectors).
*   **Pace & Sector Analysis:** Visual lap-by-lap comparison and Sector 1, 2, 3 micro-analysis.
*   **Delta Analysis:** Real-time visual gap tracking (Driver 1 vs Driver 2 time delta).
*   **Tyre Strategy Visualizer:** Automatic tyre stint detection mapped onto a visual timeline.
*   **Smooth UX:** Seamless data fetching using Streamlit status animations.

## Architecture & Clean Code

This project strictly adheres to **Clean Code** principles, modularizing logic to ensure maintainability:
- `app.py`: Contains only Streamlit UI, structured into isolated rendering functions (`render_homepage()`, `render_dashboard()`, `main()`).
- `data_loader.py`: Handles state caching, FastF1 session fetching, and API error isolation.
- `utils.py`: Centralized mathematical calculations, timedeltas processing, and formatting helpers.


---

## 🛠 Tech Stack

- **Python 3.9+**
- **FastF1** _(Telemetry & schedule data)_
- **Pandas & NumPy** _(Data manipulation & vectorization)_
- **Matplotlib** _(Data visualization)_
- **Streamlit** _(Front-end UI & routing)_

---

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/caglar-mesci/f1-data-hub.git
cd f1-data-hub

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment (Windows)
.\.venv\Scripts\activate
# Activate environment (Mac/Linux)
# source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run application
streamlit run app.py
```
