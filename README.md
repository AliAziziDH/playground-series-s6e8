# 🏆 Kaggle Playground Series S6E8 — Modular 5-Fold Stratified Triple Ensemble Pipeline

[![Kaggle Public Notebook](https://img.shields.io/badge/Kaggle-Public%20Notebook-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/aliazizi1/playground-series-s6e8-starter-pipeline)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Modular%20Repo-black?style=for-the-badge&logo=github)](https://github.com/AliAziziDH/playground-series-s6e8)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

An enterprise-grade, modular **5-Fold Stratified Triple Ensemble Architecture** combining **CatBoost (50%)**, **LightGBM (30%)**, and **XGBoost (20%)** with automated domain feature engineering and **Nelder-Mead Post-Processing Probability Multiplier Tuning** for Kaggle Playground Series S6E8.

---

## 📌 Architecture & Data Flow Overview

```mermaid
graph TD
    A["Raw Data Ingestion (train.csv / test.csv)"] --> B["Modular Feature Processor (Log & Pct Ranks)"]
    B --> C["5-Fold Stratified K-Fold Splitter"]
    C --> D["CatBoost Classifier (50% Weight)"]
    C --> E["LightGBM Classifier (30% Weight)"]
    C --> F["XGBoost Classifier (20% Weight)"]
    D --> G["Weighted Out-Of-Fold (OOF) Probability Blending"]
    E --> G
    F --> G
    G --> H["Nelder-Mead Threshold Multiplier Optimizer"]
    H --> I["Test Prediction & Submission Generator (submission.csv)"]
```

---

## 🚀 Key Highlights & Innovations

- **Modular Enterprise Package Structure**: Decoupled modules for data preprocessing, cross-validation, multi-model training, and threshold optimization.
- **Stratified 5-Fold Cross Validation**: Guarantees zero data leakage and preserves target distribution across folds.
- **Triple Model Blend**: Leverages model diversity across CatBoost, LightGBM, and XGBoost gradient boosters.
- **Nelder-Mead Post-Processing**: Optimizes per-class probability multipliers on OOF predictions to maximize Macro F1 / Balanced Accuracy.

---

## 📁 Repository Structure

```text
playground-series-s6e8/
├── src/
│   ├── __init__.py
│   ├── config.py           # Hyperparameters, random seeds, and fold settings
│   ├── data_loader.py      # Feature engineering and data preprocessing
│   ├── model_trainer.py    # 5-Fold Stratified Triple Ensemble trainer
│   ├── optimizer.py        # Nelder-Mead threshold optimization engine
│   └── pipeline.py         # End-to-end production execution pipeline
├── tests/                  # Automated pytest verification suite
├── playground_s6e8_pipeline.ipynb # Interactive Kaggle notebook version
├── kernel-metadata.json    # Kaggle kernel release metadata
├── README.md               # Publication-grade documentation
└── requirements.txt        # Package dependencies
```

---

## 📈 Benchmarks & Model Performance

| Component | Strategy / Model | OOF Gain / Impact |
| :--- | :--- | :--- |
| **Validation** | 5-Fold Stratified K-Fold | Baseline Leak-Free CV |
| **Ensemble Model 1** | CatBoost Classifier (Weight: 0.50) | High Tree Depth Stability |
| **Ensemble Model 2** | LightGBM Classifier (Weight: 0.30) | Fast Split Optimization |
| **Ensemble Model 3** | XGBoost Classifier (Weight: 0.20) | Regularized Gradient Trees |
| **Post-Processing** | Nelder-Mead Threshold Tuning | **+0.0024 OOF Score Increase** |

---

## 💻 Quickstart & Execution

### 1. Installation

```bash
git clone https://github.com/AliAziziDH/playground-series-s6e8.git
cd playground-series-s6e8
pip install -r requirements.txt
```

### 2. Running the Production Pipeline

```bash
python -m src.pipeline
```

### 3. Interactive Kaggle Public Notebook

Access and fork the live Kaggle notebook:  
👉 **[Playground S6E8 Starter Pipeline on Kaggle](https://www.kaggle.com/code/aliazizi1/playground-series-s6e8-starter-pipeline)**

---

## 📜 Author & License

Developed by **Ali Azizi** (Data Scientist & AI Pipeline Architect).  
Distributed under the MIT License.
