# Financial News-Based Stock Movement Prediction

## Overview
This project explores whether financial news headlines, combined with historical market data, can be used to predict stock price movements. The objective is to understand how much predictive signal exists in news data and how it can be combined with time-series features for improved performance.

---
## Problem Statement
Given:
* Financial news headlines
* Historical return-based features
Predict whether the stock return is above or below a chosen threshold (binary classification).
---
## Dataset
The dataset contains:
* `Date` — trading day
* `Headlines` — financial news text
* `Return_Lag_1 ... Return_Lag_100` — past return values
* `target` — actual return
---

## Approach
The project was developed in multiple stages.

### 1. Baseline Model
* Text preprocessing using lemmatization
* TF-IDF feature extraction
* Models: Logistic Regression, Linear SVM

Performance:
* Accuracy around 52–54%
This showed that text alone provides limited predictive power.
---

### 2. Transformer-Based Model (FinBERT)
* Used FinBERT embeddings for text representation
* Attempted fine-tuning

Observation:
* Fine-tuning led to overfitting
* No significant improvement over TF-IDF
This indicated that model complexity alone does not solve the problem.

---

### 3. Hybrid Model (Final Approach)

Combined:
* FinBERT embeddings (frozen)
* Lag-based numerical features
* Binary indicator for absence of news

Model:
* XGBoost classifier

Performance:
* Accuracy around 55–60%
* ROC-AUC around 0.6

This approach performed best, showing the importance of combining text with market features.

---

## Target Definition
Two approaches were tested:

1. **70th percentile split**
   * Higher accuracy (~78%)
   * Strong class imbalance

2. **Median split (final choice)**
   * Balanced dataset
   * More reliable evaluation

The median-based split was used in the final model.

---

## Threshold Selection
Using the default classification threshold (0.5) resulted in poor recall for the positive class.
The threshold was tuned using F1-score, and a value of **0.3** was selected to improve class balance.

---

## Evaluation

Metrics used:
* Accuracy (interpreted carefully)
* ROC-AUC
* F1-score
* Confusion matrix
---

## Key Findings
* Financial prediction is highly noisy with weak signal
* Accuracy can be misleading in imbalanced datasets
* Threshold selection significantly affects results
* Hybrid models (text + numerical features) outperform text-only models
* Transformer fine-tuning is prone to overfitting on small datasets

---

## How to Run
Install dependencies:
```
pip install -r requirements.txt
```
Run the model:
```
python main.py
```

---
## Technologies Used

* Python
* scikit-learn
* XGBoost
* HuggingFace Transformers (FinBERT)
* PyTorch
* Pandas, NumPy

---

## Conclusion
The project demonstrates that while financial news contains some predictive signal, it is weak and requires careful modeling. Combining textual and numerical features improves performance, but gains remain limited due to the inherent noise in financial markets.

---

## Author
Prashay Joon
