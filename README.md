# 🛰️ GeoSentinel – Real-Time Geo Intelligence System

## 🚀 Overview

GeoSentinel is a **fault-tolerant geospatial intelligence system** that tracks global aircraft activity, detects anomalies, and provides real-time, regional, and temporal insights through an interactive dashboard.

Built in under 24 hours, this project goes beyond traditional data science workflows by combining **data engineering, anomaly detection, geospatial visualization, and system reliability**.

---

## 🔥 Key Features

### 🌍 Real-Time & Simulated Tracking

* Live aircraft tracking using public aviation data
* Intelligent fallback system:

  * API → Cache → Simulation
* Ensures **zero downtime**

---

### 🚨 Anomaly Detection Engine

Detects abnormal aircraft behavior:

* High Speed (> 900 km/h)
* Low Altitude (< 1000 m)
* Suspicious Hovering

---

### 🌍 Regional Intelligence

* Continent-based filtering (Asia, Europe, etc.)
* Region-specific insights:

  * Aircraft density
  * Anomaly distribution
  * Dominant anomaly type

---

### 📈 Temporal Intelligence

* Tracks last few data snapshots
* Visualizes:

  * Aircraft trends
  * Anomaly trends

---

### 🧠 Threat Scoring System

* Assigns risk scores to aircraft:

  * Hovering → High Risk
  * High Speed → Medium Risk
  * Low Altitude → Moderate Risk

* Displays **Top high-risk aircraft**

---

### 🔥 Advanced Visualization

* Interactive geospatial dashboard
* Heatmaps for density analysis
* DBSCAN clustering for hotspot detection
* Real-time alert panel

---

### 🤖 AI Intelligence Insights

* Generates rule-based insights like:

  * High anomaly concentration
  * Cluster detection
  * High-speed trends
* Mimics an **intelligence analyst system**

---

## 🧠 System Architecture

```
Live API (OpenSky)
        ↓
     Cache Layer
        ↓
 Simulation Engine
        ↓
 Data Processing
        ↓
 Anomaly Detection
        ↓
 Intelligence Engine
        ↓
 Streamlit Dashboard (Map + Insights)
```

---

## ⚙️ Tech Stack

* **Python**
* **Streamlit** (UI Dashboard)
* **Pandas & NumPy** (Data Processing)
* **Folium & OpenStreetMap** (Geospatial Visualization)
* **Scikit-learn (DBSCAN)** (Clustering)
* **Requests** (API Handling)

---

## 🚧 Challenges & Solutions

### ⚠️ Problem: API Rate Limiting (HTTP 429)

* Frequent requests caused API failures

### ✅ Solution:

* Implemented caching layer (reduces API calls)
* Designed simulation engine (fallback data generation)
* Built multi-source pipeline:

```
API → Cache → Simulation
```

👉 Result: **System remains operational even without live data**

---

## 📸 Screenshots

> Add your screenshots here:

* 🌍 Global Dashboard
* 🚨 Threat Intelligence Panel
* 🌍 Regional View (e.g., Asia)
* 📈 Temporal Trends

---

## 🚀 Run Locally

```bash
git clone <your-repo-link>
cd GeoSentinel
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Project Highlights

* Built in **24 hours**
* Combines **data science + system design**
* Handles **real-world constraints (API limits)**
* Focus on **resilience and intelligence**

---

## 🔮 Future Improvements

* Real-time streaming using WebSockets
* Predictive trajectory modeling
* Integration with satellite data
* AI-based anomaly explanation (LLMs)

---

## 👨‍💻 Author

**Sahil Nitin Mujumdar**
Data Science Student | Cybersecurity Enthusiast

* GitHub: https://github.com/MujumdarSahil
* LinkedIn: https://www.linkedin.com/in/sahil-mujumdar-2ba179268/

---

## ⭐ If you found this interesting

Give it a ⭐ on GitHub and feel free to connect!
