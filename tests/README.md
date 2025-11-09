# 🧪 AI Lactate Advisor - Test Suite

This folder contains lightweight utilities to **verify model integrity and feature logic**
before deploying the AI Lactate Advisor app or retraining models.

---

## 📁 Contents

| File | Description |
|------|--------------|
| `model_utils.py` | Core helper module used in `app.py` and training notebooks. Handles feature prep, rolling stats, slopes, predictions, and SHAP explanations (cached per model). |
| `test_model_utils.ipynb` | Self-contained Jupyter/Colab notebook that tests every function in `model_utils.py` and includes a SHAP demo using a dummy LightGBM model. |

---

## ✅ Quick Test Instructions

1. **Open the notebook**
   ```bash
   jupyter notebook test_model_utils.ipynb
   ```
   or in Google Colab:  
   [Upload → Open in Colab → Run All]

2. **Verify output**
   You should see these key messages:
   ```
   ✅ All base utility functions executed successfully!
   ✅ SHAP explainer ran successfully (cached per model)
   ```

3. **Purpose**
   - Ensures `add_hr_slopes`, `add_rolling_features`, and `prepare_features` behave as expected.
   - Confirms `get_shap_summary()` works and caching functions correctly.
   - Tests for compatibility with both the `app.py` and model training notebooks.

---

## 🧩 Notes for Contributors

- The SHAP test uses **synthetic HR/power data**, not real athlete data.
- LightGBM is required for the SHAP demo:
  ```bash
  pip install lightgbm shap
  ```
- Do **not** modify this file when adjusting Streamlit or model logic; treat it as a regression test.

---

## 📬 Maintainer Contact

This project and supporting code are owned by **Indars Šparniņš**.  
For collaboration or reuse, please contact: **sparnins@hotmail.com**

© 2025 AI Lactate Advisor | All rights reserved
