# Startup-Valuation-and-Exit-Prediction-System
# Predictive Modeling for Startup Valuation and Success
**Course:** EEE 485 - Statistical Learning and Data Analytics
**Team:** Mehreen Irfan & Şebnem Sarı 

## 📌 Overview
This repository contains a custom-built Machine Learning pipeline designed to predict a startup's financial valuation and potential exit status (Private, Acquired, or IPO).To deeply understand the underlying algorithms, all regression and classification models were implemented entirely from scratch using Python's NumPy library without relying on pre-built ML frameworks. 

The dataset presented significant real-world challenges, including highly non-linear relationships and severe class imbalance (e.g., 75% of the startups in the dataset remained private). The project details our strategic data preprocessing pivots, such as implementing One-Hot Encoding and feature reclassification, to extract reliable predictive signals from noisy data.

## 🚀 Key Results
**Valuation Prediction (Regression):** Among our scratch-built OLS, Ridge, and Lasso regressors, the **Lasso Regressor** performed best.It effectively utilized feature selection to filter out irrelevant noise, achieving a Root Mean Square Error (RMSE) of 143.45M USD.
**Exit Status (Classification):** We successfully mitigated the "Accuracy Paradox". While powerful ensemble trees (XGBoost, AdaBoost) achieved high global accuracy by blindly predicting the majority "Private" class, our **Gaussian Naive Bayes (GNB)** model was selected as the superior classifier. Its high-bias statistical approach allowed it to accurately identify the rare, high-value exits (IPOs and Acquisitions) that other models completely ignored.

## 🛠️ Tech Stack
**Language:** Python
**Libraries:** NumPy (for zero-dependency algorithmic implementations of OLS, Ridge, Lasso, XGBoost, AdaBoost, and GNB)

## 📄 Full Methodology
For a comprehensive breakdown of the closed-form solutions, gradient descent updates, parameter tuning graphs, and per-class confusion matrices, please refer to our full **[Final Report (PDF)](Report.pdf)** included in this repository.
