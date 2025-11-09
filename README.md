# 🧠 AI-Lactate-Advisor — Quick Start
**AI-Lactate-Advisor** is an AI-powered tool for endurance athletes and coaches.  
It predicts lactate buildup and recovery readiness using wearable and biomarker data, with built-in model explainability via SHAP visualizations.
A smart endurance-training assistant that predicts **blood lactate** and **recovery readiness** using athlete data and lab biomarkers.  
Built for high-performance coaches and athletes who want **real-time metabolic insights**.

---

## 🚀 Get Started in 1 Minute

1️⃣ **Clone the repo**
```bash
git clone https://github.com/indarss/AI-Lactate-Advisor.git
cd AI-Lactate-Advisor
```

2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Run the Streamlit app**
```bash
streamlit run app.py
```

4️⃣ **To train models**
Open the notebook:
```
notebooks/AI_Lactate_Training_AutoRetrain_Versioned_Changelog_Visual_Notes.ipynb
```
Run all cells in **Google Colab** to:
- Detect and merge new lab datasets  
- Retrain models only if needed  
- Log results and metrics  
- Sync updates to GitHub automatically  

---

## 📊 Real-Time Insights

- **AI Lactate Prediction** — instant lactate-level feedback from wearable data  
- **Recovery Score** — integrates blood biomarkers + training load  
- **SHAP Visuals** — transparent model explanations  
- **Trend Dashboard** — see R² and MAE evolution over time  

---

## 🧾 Model Versioning

Every retrain creates timestamped models:
```
models/lactate_lightgbm_model_YYYY_MM_DD_HHMM.joblib
models/recovery_lightgbm_model_YYYY_MM_DD_HHMM.joblib
```
and records metadata in:
```
data/model_changelog.csv
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push your repo to GitHub  
2. In Streamlit Cloud → “New app” → choose this repo  
3. Path: `app.py`  
4. Add secret: `GITHUB_TOKEN` = _your GitHub PAT_  
5. Deploy ✅  

---

## 🧬 Contact & License

Developed by **Indars and team**.  
To use or extend this project, contact 📧 `sparnins@hotmail.com`.  
All rights reserved © 2025.

---

## 🧠 Understanding the SHAP Visuals

**What you’re seeing:**

1. **Global SHAP Importance (Top Chart)**
   - Shows which physiological or performance features (like *power, HR, slope of HR*) have the **strongest average influence** on lactate prediction across all sessions.
   - The longer the bar, the more the model relies on that signal to understand your metabolic state.
   - *Example:* “Power_mean_30s” being dominant means the model strongly associates recent power output with lactate build-up.

2. **Per-Sample SHAP Impact (Bottom Chart)**
   - Explains the model’s decision for the **latest window** of data.
   - **Green bars** = factors that **increase** predicted lactate (push toward fatigue).
   - **Red bars** = factors that **reduce** predicted lactate (indicate recovery or aerobic stability).
   - The **bar length** shows *how much* each feature contributes — longer = stronger effect.
   - *Example:* A large green “HR_slope_30s” bar means a rapidly rising HR is pushing lactate prediction upward — the athlete is nearing threshold.

3. **Interpretation for coaches:**
   - Quickly identify *why* the athlete’s lactate rose — was it power, HR drift, or instability?
   - Use it to adjust pacing or recovery cues in real time.
   - It turns a “black box” AI into a **transparent assistant** explaining its reasoning.

---

## 🧬 Recovery Dashboard

The **Recovery Dashboard** extends the AI Lactate Advisor beyond momentary fatigue analysis.
It uses **lab biomarkers** (CK, Cortisol, T/C ratio, hsCRP, Glucose, RBC) and **wearable data**
to predict an athlete's *readiness score (0–100)* for optimal training timing.

- **80–100** → 🟢 Fully recovered, safe for high-intensity sessions  
- **60–80** → 🟡 Moderately recovered, active recovery recommended  
- **Below 60** → 🔴 Rest advised before next major workout  

This feature integrates both **real-time physiological trends** and **biochemical recovery data** to form a holistic readiness indicator.

---

## 🚀 Features

- 🩸 **Lactate Model** — Predicts lactate concentration from wearable data  
- 🧬 **Recovery Model** — Estimates recovery score using biomarkers  
- 💡 **Explainability** — SHAP visualizations to interpret model decisions  
- ☁️ **Streamlit Cloud App** — Interactive dashboard  
- 🔁 **Auto-GitHub Sync** — Automatically uploads trained models  
- 🔄 **Streamlit Redeploy Trigger** — Automatically refreshes the app  

---

## 🗂️ Repository Structure
```
AI-Lactate-Advisor/
├── app.py
├── README.md
├── requirements.txt
├── models/
│   ├── lactate_lightgbm_model.joblib
│   └── recovery_lightgbm_model.joblib
├── data/
│   ├── athlete_training_dataset_1000.csv
│   └── athlete_training_dataset_with_biomarkers.csv
├── assets/
│   ├── logo.png
│   └── favicon.png
├── notebooks/
│   └── AI_Lactate_Training_Complete_Merged.ipynb
```

---

## 🔐 GitHub Token Setup

1. Go to [GitHub → Settings → Developer settings → Personal Access Tokens](https://github.com/settings/tokens)
2. Generate a **fine-grained token** for this repo with **Read/Write Contents** access
3. In **Google Colab**, go to:
   **Runtime → Manage sessions → Secrets → New Secret**
   ```
   Name: GITHUB_TOKEN
   Value: <your_personal_access_token>
   ```

---

## ☁️ Streamlit Deployment

1. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
2. Choose **New app → From GitHub**
3. Set main file path as `app.py`
4. Add secret `GITHUB_TOKEN` in the Streamlit Cloud secrets panel

---

## 💬 Credits

Developed by **Indars**  
AI-driven performance insights for endurance athletes.

