README_SHAP.txt
====================

🧠 Understanding the SHAP Visuals
---------------------------------

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

---------------------------------
Add this explanation to your hackathon presentation so coaches and judges can easily understand how AI interprets athlete physiology.

---

## 🧬 Recovery Dashboard

The **Recovery Dashboard** extends the AI Lactate Advisor beyond momentary fatigue analysis.
It uses **lab biomarkers** (CK, Cortisol, T/C ratio, hsCRP, Glucose, RBC) and **wearable data**
to predict an athlete's *readiness score (0–100)* for optimal training timing.

- **80–100** → 🟢 Fully recovered, safe for high-intensity sessions  
- **60–80** → 🟡 Moderately recovered, active recovery recommended  
- **Below 60** → 🔴 Rest advised before next major workout  

This feature integrates both **real-time physiological trends** and **biochemical recovery data** to form a holistic readiness indicator.
