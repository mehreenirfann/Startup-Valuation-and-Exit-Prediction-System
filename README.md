# Startup-Valuation-and-Exit-Prediction-System
# Predictive Modeling for Startup Valuation and Success
[cite_start]**Course:** EEE 485 - Statistical Learning and Data Analytics [cite: 1]  
[cite_start]**Team:** Mehreen Irfan & Şebnem Sarı [cite: 4]  

## 📌 Overview
[cite_start]This repository contains a custom-built Machine Learning pipeline designed to predict a startup's financial valuation and potential exit status (Private, Acquired, or IPO)[cite: 10]. [cite_start]To deeply understand the underlying algorithms, all regression and classification models were implemented entirely from scratch using Python's NumPy library without relying on pre-built ML frameworks[cite: 164, 220]. 

[cite_start]The dataset presented significant real-world challenges, including highly non-linear relationships and severe class imbalance (e.g., 75% of the startups in the dataset remained private)[cite: 66, 309]. [cite_start]The project details our strategic data preprocessing pivots, such as implementing One-Hot Encoding and feature reclassification, to extract reliable predictive signals from noisy data[cite: 157, 160, 435].

## 🚀 Key Results
* [cite_start]**Valuation Prediction (Regression):** Among our scratch-built OLS, Ridge, and Lasso regressors, the **Lasso Regressor** performed best[cite: 438, 439]. [cite_start]It effectively utilized feature selection to filter out irrelevant noise, achieving a Root Mean Square Error (RMSE) of 143.45M USD[cite: 259, 439].
* [cite_start]**Exit Status (Classification):** We successfully mitigated the "Accuracy Paradox"[cite: 310]. [cite_start]While powerful ensemble trees (XGBoost, AdaBoost) achieved high global accuracy by blindly predicting the majority "Private" class, our **Gaussian Naive Bayes (GNB)** model was selected as the superior classifier [cite: 421-426]. [cite_start]Its high-bias statistical approach allowed it to accurately identify the rare, high-value exits (IPOs and Acquisitions) that other models completely ignored[cite: 424, 442].

## 🛠️ Tech Stack
* [cite_start]**Language:** Python [cite: 164]
* [cite_start]**Libraries:** NumPy (for zero-dependency algorithmic implementations of OLS, Ridge, Lasso, XGBoost, AdaBoost, and GNB) [cite: 39, 164, 220]

## 📄 Full Methodology
[cite_start]For a comprehensive breakdown of the closed-form solutions, gradient descent updates, parameter tuning graphs, and per-class confusion matrices, please refer to our full **[Final Report (PDF)](Report.pdf)** included in this repository[cite: 6].
