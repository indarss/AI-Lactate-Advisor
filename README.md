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

## 🧩 How the Recovery Index Works

The **Recovery Index (0–100)** summarizes how ready or recovered an athlete is after a workout, blending **biochemical**, **physiological**, and **load-based** data into one clear score.

| Range | Status | Meaning | Recommendation |
|--------|--------|----------|----------------|
| 🟢 **80–100** | High Recovery | Nervous and muscular systems fully recovered. | Safe for high-intensity or competition efforts. |
| 🟡 **60–80** | Moderate Recovery | Mild residual fatigue; body still adapting. | Aerobic or endurance sessions recommended. |
| 🟠 **40–60** | Low Recovery | Noticeable strain in biomarkers and HR trend. | Restrict to low-intensity or technique sessions. |
| 🔴 **<40** | Poor Recovery | Elevated stress, insufficient regeneration. | Rest or active recovery only. |

### ⚙️ How It’s Computed
The recovery index combines key biomarker signals and model outputs:

\\[
\\text{Recovery Index} = 100 - (w_1 \\cdot CK_z + w_2 \\cdot Cortisol_z + w_3 \\cdot hsCRP_z - w_4 \\cdot T/C_z)
\\]

- **CK** – muscle damage indicator  
- **Cortisol** – hormonal stress marker  
- **hsCRP** – inflammation response  
- **T/C Ratio** – anabolic vs catabolic balance  

Each variable is normalized (z-scored) and weighted by its learned model importance.  
The final value is clipped to **0–100**, making it intuitive and actionable.

### 🧠 Interpretation Example
> **Post-session Recovery Index: 78/100 (Moderate Recovery)**  
> Indicates healthy adaptation but mild residual fatigue — athlete can train again within 12–18 hours at submaximal intensity.

In essence, the Recovery Index translates complex biomarker trends into **a simple readiness metric** that coaches and athletes can track daily.

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
## 🎛️ 3D Lactate Visualization (New Feature)

The 3D Lactate Visualization tab provides an interactive, tri-dimensional view of the athlete’s physiological state during a workout.
It helps athletes and coaches visually explore how heart rate, power, and predicted lactate interact in real time.

📡 What It Shows

The interactive 3D plot displays:
X-axis: Power (W)
Y-axis: Heart Rate (bpm)
Z-axis: Predicted Lactate (mmol/L)

Each point represents a moment in the session, color-coded by metabolic intensity:

🟦 Blue — Aerobic (stable, low lactate)
🟧 Orange — Threshold approaching (moderate lactate rise)
🔴 Red — Anaerobic (rapid lactate accumulation)

This lets users see not only when they approached threshold but how their physiological trajectory evolved.

🔍 Why It’s Useful

Traditional 2D plots show lactate OR power OR heart rate.
This feature reveals the full metabolic landscape, enabling:

📈 Identification of threshold “zones”
🧭 Analysis of pacing strategies
🔁 Detection of cardiac drift (HR rising while power stays constant)
🧠 Understanding effort–lactate relationships visually
🎓 Clear teaching/demonstration for coaches and judges

It helps athletes understand why threshold was crossed, not just that it happened.

The app:
Computes rolling and slope features from wearable data
Predicts lactate for each time window
Builds a 3D Plotly scatter surface
Applies metabolic zone colors
Renders the plot in a fully rotatable, zoomable view inside Streamlit

🚀 How to Use It

Upload a session CSV or generate a synthetic demo under Live Session
Open 🎛️ 3D Lactate Visualization
Drag, rotate, zoom, and explore your metabolic profile
Use it alongside SHAP and Recovery Dashboard for complete insight


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

---

## 🧬 Data Visualization Tools

To help athletes, coaches, and analysts better understand physiological recovery patterns, we provide a visualization notebook using **synthetic or real biomarker datasets**.

### 📓 Notebook: `plot_sample_biomarkers.ipynb`
This notebook demonstrates how to visualize biomarker–recovery interactions from the AI Lactate Advisor dataset.

**Features:**
- Auto-loads `athlete_training_dataset_with_biomarkers_SAMPLE.csv`
- Static and interactive visualizations:
  - CK, Cortisol, and Recovery trends over time  
  - Recovery vs Cortisol & CK scatter relationship  
  - Correlation heatmap between biomarkers and recovery score  
- Generates **interactive Plotly HTML files** in `/content/plots/` for sharing or embedding.

**How to Run:**
1. Open the notebook in **Google Colab** or Jupyter.
2. Upload or link your dataset in `data/`.
3. Run all cells — you’ll see inline Matplotlib plots and exported interactive Plotly dashboards.
4. Use generated charts to explore how different biomarkers influence athlete recovery.

**Example Output:**
```
✅ Saved interactive plot to /content/plots/ck_cortisol_recovery.html
✅ Saved interactive plot to /content/plots/recovery_vs_cortisol_ck.html
✅ Saved interactive plot to /content/plots/correlation_heatmap.html
```
