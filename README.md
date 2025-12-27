# Satellite-Based Crop Identification 🛰️🌱

## 📖 Project Overview
This repository hosts a machine learning solution for identifying crop types (Rubber, Oil Palm, Cocoa) using Sentinel-2 satellite imagery. Developed for the Zindi Côte d’Ivoire Byte-Sized Agriculture Challenge, the pipeline leverages **multi-temporal geospatial data**, **feature engineering**, and **CatBoost** to classify agricultural fields with high accuracy.

The solution is designed to handle seasonality and spectral similarity between tree crops by utilizing a rigorous **Stratified Group K-Fold** validation strategy and a **Grand Consensus** voting ensemble.

## 🛠️ Tech Stack
*   **Language:** Python 3.11.13
*   **Geospatial Processing:** `rasterio==1.4.4` (Satellite image I/O)
*   **Data Manipulation:** `pandas==2.2.3`, `numpy==1.26.4`
*   **Machine Learning:** `catBoost==1.2.8` (Gradient Boosting), `scikit-learn==1.2.2`
*   **Optimization:** `optuna==4.5.0` (Hyperparameter Tuning)
*   **Feature Selection:** `eli5==0.13.0` (Permutation Importance)

## 📂 Dataset Structure
The dataset consists of Sentinel-2 satellite imagery and CSV metadata.
*   **Training Set:** ~7,400 images across 12 months.
*   **Test Set:** ~2,200 images across 12 months.
*   **Targets:**
    *   `Rubber` (Class 3) - *Majority Class*
    *   `Oil Palm` (Class 2)
    *   `Cocoa` (Class 1) - *Minority Class*

The data assumes a directory structure where images are stored in `S2Images/train` and `S2Images/test`, referenced by a CSV file containing `ID`, `tifPath`, and `Target` labels.

## ⚙️ Solution Architecture

### 1. Data Preprocessing & Alignment
*   **Path Correction:** Relative paths in the CSVs are re-rooted to match the local environment.
*   **Temporal Strategy:** To ensure the model generalizes well, training data was restricted to the **6 months** present in the test set (Jan, Feb, Mar, Apr, Nov, Dec). This prevents the model from learning seasonal patterns (like rainy season cloud cover) that don't exist in the inference period.
*   **ID Reconstruction:** Since the dataset contains multiple monthly observations for the same field, `ID`s were reconstructed using the January baseline to ensure strict row-to-row correspondence across time.

### 2. Feature Engineering (The "Feature Cube")
We extract a rich set of spectral and textural features for every image tile:
*   **Spectral Indices:** Calculated 12 vegetation indices, with a focus on **Red-Edge** bands (e.g., `NDRE`, `RECI`, `MCARI`) to distinguish dense tree canopies.
*   **Band Ratios:** Generated pairwise ratios between all bands and indices to capture non-linear spectral relationships.
*   **Statistical Aggregation:** For every band and ratio, we compute **14 statistics** (Mean, Max, Skew, Kurtosis, Signal-to-Noise, etc.), transforming the 2D image into a tabular feature vector.
*   **Dimensionality Reduction:** applied **Recursive Feature Elimination (RFE)** using `eli5` to reduce thousands of generated features down to the top **128 most predictive features**.

### 3. Model Training
*   **Algorithm:** **CatBoostClassifier** (MultiClass).
*   **Validation:** **Stratified Group K-Fold (5 Splits)**.
    *   *Grouping:* Grouped by Field `ID` to prevent data leakage (ensuring all months for a specific field stay in the same fold).
    *   *Stratification:* Balanced by `Target` + `Month` to preserve class and seasonal distribution.
*   **Hyperparameters:** Tuned via **Optuna** to maximize F1-Score (Found optimal Depth=8, LR=~0.118).

### 4. Inference & Ensemble Strategy
The final prediction uses a **Spatio-Temporal "Grand Vote"**:
1.  **Multi-Model:** 5 models (from the 5 CV folds) predict on the test set.
2.  **Multi-Temporal:** Predictions are generated for all 6 available months for every field.
3.  **Aggregation:** We compile **30 votes per field** (5 folds $\times$ 6 months) and select the final class using a **Hard Voting (Mode)** mechanism.

> **Result:** This approach minimizes the risk of misclassification due to cloud cover or temporary anomalies in any single image.

## 🚀 Getting Started

### Prerequisites
```bash
pip install rasterio==1.4.4 pandas==2.2.3 numpy==1.26.4 matplotlib==3.7.2 scipy==1.15.3 scikit-learn==1.2.2 catboost==1.2.8
```

### Running the Code
1.  **Configure Paths:** Update `BASE_DIR` in the first cell to point to your dataset location.
2.  **Run Notebook:** Execute cells sequentially.
    *   *Note:* The feature extraction process is computationally intensive. Ensure you have sufficient RAM (>16GB recommended).
3.  **Output:** The final submission file will be generated as `submission.csv`.

## 📊 Performance
*   **Training F1-Score:** 0.99987 (High capacity)
*   **Cross-Validation F1-Score:** ~0.89147
*   **Test Class Distribution:** Rubber (~42%), Oil Palm (~38%), Cocoa (~20%).

---
*Created by Isaac Oluwafemi Ogunniyi for the Zindi Côte d’Ivoire Byte-Sized Agriculture Challenge.*
