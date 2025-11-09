# 🧠 AI-Lactate-Advisor Setup Guide

This guide explains how to run and maintain your **AI-Lactate-Advisor** project, including dataset updates, model retraining, and Streamlit deployment.

---

## ⚙️ 1️⃣ Prerequisites

Before you begin, make sure you have:

- **Google Colab** (for model training)
- **GitHub repository** (e.g. `indarss/AI-Lactate-Advisor`)
- **Streamlit Cloud account**
- A **GitHub Personal Access Token (PAT)** with access to your repo

---

## 🧩 2️⃣ Repository Structure

Your repository should look like this:

```
AI-Lactate-Advisor/
├── app.py
├── model_utils.py
├── requirements.txt
├── README.md
├── data/
│   ├── athlete_training_dataset_with_biomarkers.csv
│   ├── model_changelog.csv
│   └── new_lab_data_2025_11.csv
├── models/
│   ├── lactate_lightgbm_model.joblib
│   ├── recovery_lightgbm_model.joblib
│   ├── lactate_lightgbm_model_2025_11_08_1530.joblib
│   └── recovery_lightgbm_model_2025_11_08_1530.joblib
└── notebooks/
    └── AI_Lactate_Training_AutoRetrain_Versioned_Changelog_Visual_Notes.ipynb
```

---

## 🚀 3️⃣ Running the Notebook in Google Colab

1. Upload the `AI_Lactate_Training_AutoRetrain_Versioned_Changelog_Visual_Notes.ipynb` file to your **Google Drive** or open it directly in Colab.
2. Mount your project folder (if stored in Drive) or upload your dataset files manually.
3. In Colab, go to:
   **Runtime → Manage Sessions → Secrets → Add New Secret**
   - Name: `GITHUB_TOKEN`
   - Value: your GitHub Personal Access Token (PAT)
4. Run all cells top-to-bottom.

✅ The notebook will:
- Detect new lab datasets in `/data/`
- Merge and update them automatically
- Retrain models **only if needed**
- Save both **versioned and latest model files**
- Log results to `model_changelog.csv`
- Upload everything to GitHub automatically

---

## 🧾 4️⃣ Model Versioning & Changelog

Each training creates new model versions like:

```
models/lactate_lightgbm_model_2025_11_08_1530.joblib
models/recovery_lightgbm_model_2025_11_08_1530.joblib
```

and logs their performance in:

```
data/model_changelog.csv
```

You can visualize model progress over time directly in the notebook (R² and MAE trends).

---

## ☁️ 5️⃣ Deploying to Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud).
2. Create a new app → Select your GitHub repo (`indarss/AI-Lactate-Advisor`).
3. Set **Main file path** as:
   ```
   app.py
   ```
4. Add the secret in Streamlit Cloud:
   - **Name:** `GITHUB_TOKEN`
   - **Value:** your GitHub PAT
5. Deploy!

Streamlit will install dependencies from `requirements.txt` and run `app.py` automatically.

---

## 🧬 6️⃣ Updating Models with New Lab Data

When new lab datasets (e.g., `lab_feb_2025.csv`) are available:

1. Add the file to your local `/data/` folder or upload it in Colab.
2. Run the notebook again — it will detect, merge, retrain, and push updates automatically.
3. The Streamlit app will use the latest `.joblib` models once redeployed.

---

## 🧠 7️⃣ Notes and Best Practices

- **Retraining frequency:** once per new lab batch (e.g., monthly)
- **Always keep** both the timestamped and current `.joblib` files
- **Commit often:** ensures version safety for both data and models
- **Changelog helps** explain performance evolution (great for hackathons!)

---

📧 For questions or permissions, contact **Indars** at `sparnins@hotmail.com`.

