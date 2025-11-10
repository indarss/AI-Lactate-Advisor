# 🧠 AI Lactate Advisor
[![Model Utilities Test](https://github.com/indarss/AI-Lactate-Advisor/actions/workflows/test_utils.yml/badge.svg)](https://github.com/indarss/AI-Lactate-Advisor/actions/workflows/test_utils.yml)

**AI-Lactate-Advisor** is an AI-powered assistant for endurance athletes and coaches.  
It predicts **blood lactate** and **recovery readiness** using wearable and lab data — enhanced with **explainable AI** (SHAP) to make insights transparent and actionable.

---

## 🚀 Quick Start

### 1️⃣ Clone and Setup
```bash
git clone https://github.com/indarss/AI-Lactate-Advisor.git
cd AI-Lactate-Advisor
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

### 3️⃣ (Optional) Train or Update Models
Open in **Google Colab**:
```
notebooks/AI_Lactate_Training_AutoRetrain_Versioned.ipynb
```
This notebook:
- Detects & merges new lab data  
- Retrains only if needed  
- Logs metrics and updates GitHub automatically  

---

## 📊 Core Features

| Feature | Description |
|----------|--------------|
| 🩸 **Lactate Model** | Predicts lactate buildup from HR, power, and pace signals |
| 🧬 **Recovery Model** | Estimates readiness score (0–100) using biomarkers |
| 💡 **Explainability** | SHAP visualizations to interpret every prediction |
| 💓 **Trend Dashboard** | Live HR slope and power trends with Plotly |
| ☁️ **Streamlit Cloud App** | Real-time interactive dashboard |
| 🔁 **Auto Model Versioning** | Saves timestamped `.joblib` models with changelog |
| ⚙️ **CI Testing** | GitHub Actions badge validates model utils & SHAP logic |

---

## 🧠 Understanding SHAP Visuals

### 1. Global SHAP Importance
Shows the features that most influence lactate prediction.
- Longer bars = stronger global impact.
- Example: high *power_mean_30s* means intensity drives lactate buildup.

### 2. Per-Sample SHAP Impact
Explains the **latest prediction window**.
- 🟩 Green → Increases lactate (fatigue signal)  
- 🟥 Red → Reduces lactate (recovery trend)

### 3. Coaching Use
Instantly see *why* lactate rose — due to HR drift, power, or instability.  
Helps optimize pacing, intervals, and recovery.

---

## 🧬 Recovery Dashboard

Predicts **readiness** (0–100) combining biomarkers and load metrics.

| Score | Status | Recommendation |
|-------|---------|----------------|
| 🟢 80–100 | Fully recovered | Safe for intensity |
| 🟡 60–80 | Moderate | Active recovery |
| 🔴 <60 | Fatigued | Rest advised |

---

## 🗂️ Repository Layout

```
AI-Lactate-Advisor/
├── app.py
├── model_utils.py
├── models/
│   ├── lactate_lightgbm_model.joblib
│   ├── recovery_lightgbm_model.joblib
├── notebooks/
│   ├── AI_Lactate_Training_AutoRetrain_Versioned.ipynb
├── tests/
│   ├── test_model_utils.ipynb
│   └── README.md
└── .github/
    └── workflows/
        └── test_utils.yml
```

---

## ☁️ Deployment (Streamlit Cloud)

1. Push repo to GitHub  
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → *New App*  
3. Select this repo and path `app.py`  
4. Add `GITHUB_TOKEN` secret (fine-grained PAT with Read/Write Contents)  
5. Click **Deploy** ✅  

---

## 🧪 Continuous Integration

Every commit triggers automatic testing via **GitHub Actions**:  
- Executes `test_model_utils.ipynb`  
- Verifies rolling stats, SHAP caching, and slope logic  
- Uploads result notebook as artifact  

You can view results under the **Actions** tab or check the badge at the top of this README.

---

## 🧭 Future Enhancements

These planned improvements will make **AI Lactate Advisor** even more powerful and practical for real-world sports environments:

| Area | Planned Feature | Description |
|------|------------------|--------------|
| ⌚ **Wearable Sync** | Real-time Bluetooth/ANT+ integration | Connects directly to heart rate monitors, power meters, or Garmin/Wahoo devices |
| 🤖 **Personalized Thresholds** | Adaptive lactate threshold model | Learns each athlete’s unique HR–power–lactate profile over time |
| ☁️ **Cloud Database** | Historical training and biomarker tracking | Enables long-term athlete profiling and overtraining alerts |
| 🧠 **AI Coaching Assistant** | Voice/chat-based feedback loop | Provides instant recovery or pacing guidance during workouts |
| 🧪 **Advanced Biomarkers** | Integrate new lab metrics | Add hormone, glucose, and HRV correlation for precision recovery readiness |
| 📈 **Performance Insights Dashboard** | Weekly summary trends | Auto-generated reports for coaches and teams |

---



---

## 🆕 Live Mode – Real-Time Wearable Data Streaming

The **Live Mode** tab extends the AI Lactate Advisor from static datasets to **real-time streaming** and **connected wearable analytics**.  
It introduces dynamic Polar API integration, mock data simulation, and upload-based session analysis.

### 🔗 Features:
- **Polar OAuth2 Integration** – authenticate and securely pull HR, power, and pace data from your Polar account.  
- **Mock Stream Mode** – simulate wearable telemetry in real time to demo or test AI predictions without devices.  
- **Upload CSV/TCX** – analyze exported workout files for lactate and recovery prediction.  
- **Plotly Live Charts** – interactive, dual-axis charts overlaying heart rate, power, and predicted lactate in real time.  

### 🧩 Technical Flow:
1. Authenticate with Polar or use the built-in Mock Stream.
2. Stream incoming HR/power data into the `make_features()` pipeline.
3. Model predicts **instantaneous lactate** and **recovery trend**.
4. Streamlit renders results via Plotly with millisecond responsiveness.

### 💻 Code Integration:
- The new **`_render_live_mode_tab()`** function is automatically loaded with the app.
- Models are reused (`lactate_lightgbm_model.joblib`, `recovery_lightgbm_model.joblib`) or gracefully skipped if missing.
- Fallback logic ensures the app remains stable even if no wearable or model files are present.

### 🖼️ Architecture Diagram
![Wearable to AI Pipeline](A_flowchart_diagram_illustrates_the_integration_pr.png)
*Figure: End-to-end wearable data to AI prediction workflow.*


## 💬 Author & License

Developed by **Indars Sparniņš** and team.  
📧 Contact: **sparnins@hotmail.com**  
All rights reserved © 2025 AI Lactate Advisor.
