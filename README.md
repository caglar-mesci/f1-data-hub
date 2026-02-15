# 🏁 F1 Data Hub

[![Live Demo](https://img.shields.io/badge/Live-Demo-red?style=for-the-badge)](https://f1-data-caglarmesci.streamlit.app/)

An interactive Formula 1 race analytics dashboard built with FastF1 and Streamlit.

This project transforms official timing data into a structured analytical interface for performance benchmarking and tyre strategy evaluation.

---

## 🌐 Live Demo

https://f1-data-caglarmesci.streamlit.app/

---

## 🚀 Features

- Season → Grand Prix → Session selection  
- Driver-to-driver performance comparison  
- Lap-by-lap pace visualization  
- Delta analysis (Driver 1 − Driver 2)  
- Automatic tyre stint detection  
- Tyre strategy timeline visualization  
- Clean time formatting (mm:ss.mmm)  
- Local caching for performance optimization  

---

## 🏆 Historical Overview

Homepage includes the most successful drivers:

- Michael Schumacher (7)
- Lewis Hamilton (7)
- Juan Manuel Fangio (5)
- Alain Prost (4)
- Sebastian Vettel (4)
- Max Verstappen (4)

---

## 🛠 Tech Stack

- Python
- FastF1
- Pandas
- NumPy
- Matplotlib
- Streamlit

---

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.\.venv\Scripts\activate

# Activate environment (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py


