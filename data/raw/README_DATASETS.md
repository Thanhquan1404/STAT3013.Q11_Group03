#  **Datasets Overview**

This repository uses **two important clinical datasets** related to liver disease research:

1. **Indian Liver Patient Dataset (ILPD)** — binary prediction
2. **Liver Cirrhosis Dataset** — multiclass cirrhosis stage classification

Both datasets appear in multiple peer-reviewed publications and are widely used in biomedical machine learning research.

#  **1. Indian Liver Patient Dataset (ILPD)**

###  **Source**

* UCI Machine Learning Repository
  https:/*/archive.ics.uci.edu/dataset/60/indian+liver+patient+dataset (remove the * when using)
* Originally collected by **Narayana Medical College and Hospital, Nellore, Andhra Pradesh, India**

###  **Reference Paper**

* **Bendi Venkata Ramana, "Diagnosing Liver Patients Using Classification Algorithms", 2012**
  https:/*/ieeexplore.ieee.org/document/6179021

###  **Dataset Description**

The ILPD dataset contains **583 patient records**, each with **10 clinical features** and **1 binary label**:

* **1 = Patient has liver disease**
* **2 = Patient does not have liver disease**

Features include:

* Age
* Gender
* Total Bilirubin
* Direct Bilirubin
* Alkaline Phosphotase
* Alanine Aminotransferase (SGPT)
* Aspartate Aminotransferase (SGOT)
* Total Protein
* Albumin
* Albumin/Globulin Ratio

###  **Clinical Meaning**

These features are standard liver function test (LFT) metrics used to detect:

* Hepatitis
* Cirrhosis
* Fatty liver disease
* Alcohol-related liver damage
* Obstruction of bile ducts

Elevations in **Bilirubin, SGOT, SGPT, AlkPhos** are strongly associated with liver dysfunction.

###  **Scientific Usage**

ILPD appears in numerous ML and medical informatics studies:

* Early comparison of classification methods for medical diagnosis
* Benchmark dataset for **logistic regression, SVM, decision trees, random forests, neural networks**
* Used in several studies on model interpretability and feature importance

---

#  **2. Liver Cirrhosis Dataset**

###  **Source**

* Mayo Clinic Trial on Primary Biliary Cirrhosis (PBC)
* Hosted at::

  * **pydataset** (“PBC” dataset)
  * **MASS package in R** (“cirrhosis” dataset)
  * **Kaggle community repostings**

###  **Reference Paper**

* **Therneau & Grambsch, "Modeling Survival Data: Extending the Cox Model" (2000)**
  https:/*/link.springer.com/book/10.1007/978-1-4757-3294-8

* **Murtaugh et al., "Primary biliary cirrhosis: prediction of short-term survival based on repeated patient visits" (1994)**
  https:/*/pubmed.ncbi.nlm.nih.gov/8117285/

###  **Dataset Description**

The liver cirrhosis dataset contains **patients diagnosed with Primary Biliary Cirrhosis**, with the goal of predicting **disease stage (1–4)**.

Common features include:

* Age
* Sex
* Ascites
* Hepatomegaly
* Spiders (vascular lesions)
* Edema
* Bilirubin
* Cholesterol
* Albumin
* Copper
* Alkaline Phosphatase
* SGOT
* Triglycerides
* Platelets
* Prothrombin time
* Clinical stage (1–4)

###  **Clinical Meaning**

The dataset captures multiple physiological and biochemical indicators:

* **Bilirubin, Albumin, Prothrombin** → markers of liver failure
* **Ascites, Spiders, Edema** → symptoms of advanced cirrhosis
* **Copper, Alk_Phos, SGOT** → cholestasis and hepatocellular injury
* **Thrombocytopenia (low platelets)** → portal hypertension

The labels (1–4) correspond to increasing clinical severity.

###  **Scientific Usage**

This dataset is extensively used in:

* Survival analysis
* Multi-class classification (Softmax Regression, XGBoost, Random Forests)
* PCA and LDA visualization
* Medical risk stratification
* Clinical decision-support research

Notably used in studies that evaluate:

* Feature importance in cirrhosis progression
* Predictive modeling for liver transplant prioritization
* Non-invasive disease staging

#  **Why These Datasets Matter**

Both datasets are:

### ✔ Clinically grounded

Collected from hospitals and medical trials using real laboratory measurements.

### ✔ Frequently used in biomedical ML

Over 50+ published papers rely on ILPD and PBC/Cirrhosis dataset variations.

### ✔ Suitable for teaching & benchmarking

They provide a good balance of:

* Missing values
* Outliers
* Class imbalance
* Nonlinear relationships

### ✔ Useful for multi-phase research

* **Phase 1:** Descriptive statistical analysis
* **Phase 2:** Regression models (Logistic, Softmax)
* **Phase 3:** Non-linear classification (KNN, SVM, Decision Tree, RF)
* **Phase 4:** Model interpretation (SHAP, PCA, PDA, ROC, F1)

#  **Suggested Placement in the Repo**

Create a folder:

```
/datasets/
   README_DATASETS.md
   indian_liver_disease.csv
   liver_cirrhosis.csv
```

And place the above text inside **README_DATASETS.md**.
