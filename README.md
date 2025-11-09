# LATS1_QSAR_Model
# QSAR Model for LATS1 Kinase Binding Affinity Prediction

This repository contains data and Python scripts for constructing a **machine learning–based Quantitative Structure–Activity Relationship (QSAR)** model to predict the **binding affinity (pIC₅₀)** of small molecules targeting **LATS1 kinase**, a key enzyme in the Hippo signaling pathway.

---

## 📂 Project Structure

```
LATS1_QSAR_Model/
├── data/
│   ├── lats1_full_descriptors.csv        # All 1826 Mordred descriptors
│   ├── lats1_selected_descriptors.csv    # ~200 selected descriptors + pIC50
├── src/
│   ├── feature_selection.py              # Feature filtering & preprocessing
│   ├── model_training.py                 # Regression model training and validation
│   ├── shap_analysis.py                  # Explainability / feature importance
├── results/
│   └── model.pkl                         # Trained model 
├── requirements.txt
└── README.md
```

---

## ⚙️ Methodology Overview

1. **Data Curation:**  
   - Ligand IC₅₀ values for LATS1 kinase curated from ChEMBL.
   - Converted to log-scale (pIC₅₀) for normalization.

2. **Descriptor Calculation:**  
   - 1826 1D, 2D, and 3D molecular descriptors generated with the `mordred` Python package.

3. **Feature Selection:**  
   - Redundant/highly correlated descriptors excluded.
   - Forward Feature Selection (FFS) reduced feature set to ~200 key descriptors.

4. **Model Development:**  
   - Multiple regressors tested: Linear, Ridge, Lasso, RF, SVR, etc.
   - **Multivariate Linear Regression** chosen for robustness and interpretability.

5. **Model Validation:**  
   - Train/test split (90/10) and jackknife resampling.
   - Model metrics:  
     - **Train:** r = 0.79, MAE = 0.53  
     - **Test:** r = 0.71, MAE = 0.56  
     - **Jackknife:** r = 0.75, MAE = 0.58

---

## 🧠 Key Insights

| Descriptor   | Type          | Biological Relevance                       |
|--------------|---------------|-------------------------------------------|
| MAXssssC     | Structural    | Molecular saturation, hydrophobicity       |
| SlogP_VSA3   | Lipophilicity | Surface area, membrane permeability        |
| SaasC        | Aromaticity   | π-π stacking, aromatic exposure            |
| AATS6se      | Electronegativity | Long-range electronic correlation    |
| MATS4dv      | Valence       | Charge distribution in binding regions     |

---

## 🚀 How to Run

**Clone the repository:**
```bash
git clone https://github.com/<your-username>/LATS1_QSAR_Model.git
cd LATS1_QSAR_Model
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run feature selection and model training:**
```bash
python src/feature_selection.py
python src/model_training.py
```

**Run SHAP analysis for feature importance:**
```bash
python src/shap_analysis.py
```

---

## 🧩 Dependencies

All required Python packages are listed in `requirements.txt`.
- pandas
- numpy
- scikit-learn
- mordred
- rdkit
- openbabel
- shap
- matplotlib
- scipy

---

## 📊 Dataset Summary

- **Source:** ChEMBL database
- **Unique ligands:** 286
- **Descriptors calculated:** 1826
- **Selected descriptors for final model:** ~200
- **Target variable:** pIC₅₀ (−log₁₀(IC₅₀))

---

## 📜 Notes

This repository is under active development.  
New models and analysis scripts will be added.  
