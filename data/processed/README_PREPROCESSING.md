# README_PREPROCESSING.md  
## Scientific Data Preprocessing Pipeline for Machine Learning

### Overview

This preprocessing pipeline implements a **reproducible, statistically grounded, and publication-ready** sequence of data preparation steps tailored for **supervised classification tasks** on tabular datasets. It addresses three fundamental challenges in real-world data analysis:

1. **Missing Data (Data Completeness)**
2. **Class Imbalance (Statistical Power & Model Bias)**
3. **Categorical Encoding (Algorithm Compatibility)**

The pipeline is designed for integration into **end-to-end machine learning workflows**, ensuring **transparency**, **reproducibility**, and **scientific rigor**.

### 1. Missing Value Analysis & Imputation

#### Technique: `print_nan_percentage(df)`
- **Purpose**: Quantify the extent of missingness per feature.
- **Output**: 
  - Absolute count of `NaN` values
  - Percentage of missing data per column
- **Scientific Justification**:
  > Missing data can introduce bias and reduce effective sample size. Systematic reporting is required before any imputation to assess data quality and inform downstream decisions (Little & Rubin, 2019).

#### Technique: `replace_nan(df, option="mean")`
- **Numeric Features**:
  - **Mean Imputation** (`option="mean"`): Preserves central tendency; suitable for symmetrically distributed data.
  - **Median Imputation** (`option="median"`): Robust to outliers; recommended for skewed distributions.
- **Categorical/Non-numeric Features**:
  - Filled with `"Unknown"` → preserves category integrity and avoids introducing spurious modes.

> **Rationale**: Univariate imputation maintains interpretability and computational efficiency. Mean/median are location estimators consistent with maximum likelihood under normality or robustness assumptions (Schafer & Graham, 2002).

### 2. Class Imbalance Correction via SMOTE

#### Technique: `apply_smote(df, target_col)`
- **Method**: **Synthetic Minority Over-sampling Technique (SMOTE)** (Chawla et al., 2002)
- **Mechanism**:
  1. One-hot encode all categorical predictors → ensures numerical input compatibility.
  2. For each minority class instance, generate synthetic examples by interpolating between the instance and its *k*-nearest neighbors in feature space.
  3. Result: balanced class distribution without duplication (unlike random oversampling).

#### Pre- and Post-SMOTE Class Distribution Reported

> **Scientific Advantage**:
> - Mitigates **bias toward majority class** in model training.
> - Improves **sensitivity (recall)** for rare events (e.g., disease detection, fraud).
> - Synthetic samples lie on the empirical data manifold, reducing overfitting risk compared to naive duplication.

> **Citation**:  
> Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
