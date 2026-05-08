# Sentiment Best Model Report

## RandomizedSearchCV for tuned TF-IDF + Linear SVM

Best tuned model

Best preprocessing: stemming
Best model: Linear SVM

Best parameters:
  tfidf__sublinear_tf: True
  tfidf__ngram_range: (1, 2)
  tfidf__min_df: 1
  tfidf__max_features: 50000
  svm__class_weight: balanced
  svm__C: 1.0

Average scores for this model:
  CV Macro F1: 0.789
  Test Macro F1: 0.801
  Weighted F1: 0.961
  Accuracy: 0.963

Precision / Recall / F1 by class:

```text
              precision    recall  f1-score   support

    negative       0.84      0.76      0.80       485
     neutral       0.70      0.55      0.62       555
    positive       0.98      0.99      0.98     11861

    accuracy                           0.96     12901
   macro avg       0.84      0.77      0.80     12901
weighted avg       0.96      0.96      0.96     12901

```

## Macro F1 for each setup

The first rows are the three tuned RandomizedSearchCV setups. The remaining rows are the fixed broad baseline setups.

| Rank | Setup type | Preprocessing | Vectorizer | Model | Macro F1 |
|---:|---|---|---|---|---:|
| 1 | Tuned RandomizedSearchCV | stemming | tfidf | Linear SVM | 0.801 |
| 2 | Tuned RandomizedSearchCV | raw_tokens | tfidf | Linear SVM | 0.801 |
| 3 | Tuned RandomizedSearchCV | lemmatization | tfidf | Linear SVM | 0.803 |
| 4 | Fixed baseline | stemming | tfidf | Linear SVM | 0.782 |
| 5 | Fixed baseline | lemmatization | tfidf | Linear SVM | 0.781 |
| 6 | Fixed baseline | raw_tokens | tfidf | Linear SVM | 0.780 |
| 7 | Fixed baseline | basic_clean | tfidf | Linear SVM | 0.779 |
| 8 | Fixed baseline | stemming | tfidf | Random Forest | 0.733 |
| 9 | Fixed baseline | raw_tokens | tfidf | Random Forest | 0.728 |
| 10 | Fixed baseline | basic_clean | tfidf | Random Forest | 0.728 |
| 11 | Fixed baseline | lemmatization | tfidf | Random Forest | 0.726 |
| 12 | Fixed baseline | lemmatization | tfidf | Logistic Regression | 0.722 |
| 13 | Fixed baseline | raw_tokens | tfidf | Logistic Regression | 0.720 |
| 14 | Fixed baseline | basic_clean | tfidf | Logistic Regression | 0.720 |
| 15 | Fixed baseline | stemming | tfidf | Logistic Regression | 0.719 |
| 16 | Fixed baseline | stemming | tfidf | XGBoost | 0.606 |
| 17 | Fixed baseline | basic_clean | tfidf | XGBoost | 0.599 |
| 18 | Fixed baseline | raw_tokens | tfidf | XGBoost | 0.595 |
| 19 | Fixed baseline | lemmatization | tfidf | XGBoost | 0.593 |
| 20 | Fixed baseline | raw_tokens | tfidf | Naive Bayes | 0.508 |
| 21 | Fixed baseline | basic_clean | tfidf | Naive Bayes | 0.507 |
| 22 | Fixed baseline | stemming | tfidf | Naive Bayes | 0.501 |
| 23 | Fixed baseline | lemmatization | tfidf | Naive Bayes | 0.501 |

## Tuned top 3 baseline setups

### 1. stemming + tfidf + Linear SVM

Average scores for this tuned model:
  CV Macro F1: 0.789
  Test Macro F1: 0.801
  Weighted F1: 0.961
  Accuracy: 0.963

### 2. raw_tokens + tfidf + Linear SVM

Average scores for this tuned model:
  CV Macro F1: 0.787
  Test Macro F1: 0.801
  Weighted F1: 0.961
  Accuracy: 0.963

### 3. lemmatization + tfidf + Linear SVM

Average scores for this tuned model:
  CV Macro F1: 0.785
  Test Macro F1: 0.803
  Weighted F1: 0.961
  Accuracy: 0.963

## Best 3 broad baseline models by CV Macro F1

### 1. stemming + tfidf + Linear SVM

Model setup:
  Cleaning / preprocessing: stemming
  Vectorizer: tfidf
  Model: Linear SVM

Average scores for this model:
  CV Macro F1: 0.782
  Test Macro F1: 0.792
  Weighted F1: 0.957
  Accuracy: 0.958

Precision / Recall / F1 by class:
  Negative: 0.826 / 0.761 / 0.792
  Neutral:  0.619 / 0.589 / 0.604
  Positive: 0.978 / 0.983 / 0.981

### 2. lemmatization + tfidf + Linear SVM

Model setup:
  Cleaning / preprocessing: lemmatization
  Vectorizer: tfidf
  Model: Linear SVM

Average scores for this model:
  CV Macro F1: 0.781
  Test Macro F1: 0.795
  Weighted F1: 0.959
  Accuracy: 0.959

Precision / Recall / F1 by class:
  Negative: 0.824 / 0.761 / 0.791
  Neutral:  0.632 / 0.591 / 0.611
  Positive: 0.979 / 0.985 / 0.982

### 3. raw_tokens + tfidf + Linear SVM

Model setup:
  Cleaning / preprocessing: raw_tokens
  Vectorizer: tfidf
  Model: Linear SVM

Average scores for this model:
  CV Macro F1: 0.780
  Test Macro F1: 0.796
  Weighted F1: 0.958
  Accuracy: 0.959

Precision / Recall / F1 by class:
  Negative: 0.814 / 0.765 / 0.789
  Neutral:  0.644 / 0.593 / 0.617
  Positive: 0.978 / 0.984 / 0.981

