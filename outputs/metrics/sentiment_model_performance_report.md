# Sentiment Model Performance Report

Models are sorted by cross-validation macro F1.

## stemming + tfidf + Linear SVM

- CV Macro F1: `0.782`
- Test Accuracy: `0.958`
- Test Macro F1: `0.792`
- Test Weighted F1: `0.957`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.826 | 0.761 | 0.792 |
| neutral | 0.619 | 0.589 | 0.604 |
| positive | 0.978 | 0.983 | 0.981 |

## lemmatization + tfidf + Linear SVM

- CV Macro F1: `0.781`
- Test Accuracy: `0.959`
- Test Macro F1: `0.795`
- Test Weighted F1: `0.959`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.824 | 0.761 | 0.791 |
| neutral | 0.632 | 0.591 | 0.611 |
| positive | 0.979 | 0.985 | 0.982 |

## raw_tokens + tfidf + Linear SVM

- CV Macro F1: `0.780`
- Test Accuracy: `0.959`
- Test Macro F1: `0.796`
- Test Weighted F1: `0.958`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.814 | 0.765 | 0.789 |
| neutral | 0.644 | 0.593 | 0.617 |
| positive | 0.978 | 0.984 | 0.981 |

## basic_clean + tfidf + Linear SVM

- CV Macro F1: `0.779`
- Test Accuracy: `0.959`
- Test Macro F1: `0.795`
- Test Weighted F1: `0.958`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.815 | 0.763 | 0.788 |
| neutral | 0.640 | 0.593 | 0.616 |
| positive | 0.978 | 0.984 | 0.981 |

## stemming + tfidf + Random Forest

- CV Macro F1: `0.733`
- Test Accuracy: `0.959`
- Test Macro F1: `0.762`
- Test Weighted F1: `0.952`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.948 | 0.563 | 0.706 |
| neutral | 0.992 | 0.431 | 0.601 |
| positive | 0.958 | 0.999 | 0.978 |

## raw_tokens + tfidf + Random Forest

- CV Macro F1: `0.728`
- Test Accuracy: `0.959`
- Test Macro F1: `0.766`
- Test Weighted F1: `0.952`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.971 | 0.555 | 0.706 |
| neutral | 1.000 | 0.441 | 0.613 |
| positive | 0.958 | 1.000 | 0.978 |

## basic_clean + tfidf + Random Forest

- CV Macro F1: `0.728`
- Test Accuracy: `0.959`
- Test Macro F1: `0.764`
- Test Weighted F1: `0.952`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.964 | 0.557 | 0.706 |
| neutral | 0.980 | 0.440 | 0.607 |
| positive | 0.958 | 0.999 | 0.978 |

## lemmatization + tfidf + Random Forest

- CV Macro F1: `0.726`
- Test Accuracy: `0.959`
- Test Macro F1: `0.764`
- Test Weighted F1: `0.952`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.968 | 0.559 | 0.708 |
| neutral | 0.976 | 0.440 | 0.606 |
| positive | 0.958 | 0.999 | 0.978 |

## lemmatization + tfidf + Logistic Regression

- CV Macro F1: `0.722`
- Test Accuracy: `0.917`
- Test Macro F1: `0.719`
- Test Weighted F1: `0.929`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.646 | 0.827 | 0.725 |
| neutral | 0.353 | 0.717 | 0.473 |
| positive | 0.989 | 0.930 | 0.958 |

## raw_tokens + tfidf + Logistic Regression

- CV Macro F1: `0.720`
- Test Accuracy: `0.920`
- Test Macro F1: `0.725`
- Test Weighted F1: `0.931`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.652 | 0.835 | 0.732 |
| neutral | 0.364 | 0.715 | 0.483 |
| positive | 0.989 | 0.933 | 0.960 |

## basic_clean + tfidf + Logistic Regression

- CV Macro F1: `0.720`
- Test Accuracy: `0.919`
- Test Macro F1: `0.724`
- Test Weighted F1: `0.930`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.650 | 0.835 | 0.731 |
| neutral | 0.362 | 0.717 | 0.481 |
| positive | 0.989 | 0.932 | 0.960 |

## stemming + tfidf + Logistic Regression

- CV Macro F1: `0.719`
- Test Accuracy: `0.915`
- Test Macro F1: `0.714`
- Test Weighted F1: `0.928`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.639 | 0.814 | 0.716 |
| neutral | 0.347 | 0.714 | 0.467 |
| positive | 0.989 | 0.929 | 0.958 |

## stemming + tfidf + XGBoost

- CV Macro F1: `0.606`
- Test Accuracy: `0.940`
- Test Macro F1: `0.606`
- Test Weighted F1: `0.925`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.879 | 0.419 | 0.567 |
| neutral | 0.817 | 0.169 | 0.281 |
| positive | 0.943 | 0.998 | 0.969 |

## basic_clean + tfidf + XGBoost

- CV Macro F1: `0.599`
- Test Accuracy: `0.940`
- Test Macro F1: `0.599`
- Test Weighted F1: `0.923`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.889 | 0.396 | 0.548 |
| neutral | 0.839 | 0.169 | 0.282 |
| positive | 0.941 | 0.998 | 0.969 |

## raw_tokens + tfidf + XGBoost

- CV Macro F1: `0.595`
- Test Accuracy: `0.939`
- Test Macro F1: `0.595`
- Test Weighted F1: `0.922`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.892 | 0.390 | 0.542 |
| neutral | 0.827 | 0.164 | 0.274 |
| positive | 0.941 | 0.998 | 0.968 |

## lemmatization + tfidf + XGBoost

- CV Macro F1: `0.593`
- Test Accuracy: `0.939`
- Test Macro F1: `0.593`
- Test Weighted F1: `0.923`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.885 | 0.381 | 0.533 |
| neutral | 0.816 | 0.168 | 0.278 |
| positive | 0.941 | 0.998 | 0.969 |

## raw_tokens + tfidf + Naive Bayes

- CV Macro F1: `0.508`
- Test Accuracy: `0.935`
- Test Macro F1: `0.541`
- Test Weighted F1: `0.914`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.820 | 0.386 | 0.525 |
| neutral | 0.907 | 0.070 | 0.130 |
| positive | 0.937 | 0.998 | 0.967 |

## basic_clean + tfidf + Naive Bayes

- CV Macro F1: `0.507`
- Test Accuracy: `0.935`
- Test Macro F1: `0.537`
- Test Weighted F1: `0.914`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.821 | 0.377 | 0.517 |
| neutral | 0.905 | 0.068 | 0.127 |
| positive | 0.937 | 0.998 | 0.967 |

## stemming + tfidf + Naive Bayes

- CV Macro F1: `0.501`
- Test Accuracy: `0.934`
- Test Macro F1: `0.531`
- Test Weighted F1: `0.913`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.836 | 0.367 | 0.510 |
| neutral | 0.814 | 0.063 | 0.117 |
| positive | 0.936 | 0.998 | 0.966 |

## lemmatization + tfidf + Naive Bayes

- CV Macro F1: `0.501`
- Test Accuracy: `0.935`
- Test Macro F1: `0.538`
- Test Weighted F1: `0.914`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.828 | 0.377 | 0.518 |
| neutral | 0.830 | 0.070 | 0.130 |
| positive | 0.937 | 0.998 | 0.967 |

# Tuned Models From RandomizedSearchCV

The top three fixed baseline setups were tuned. They are sorted by CV Macro F1.

| Rank | Preprocessing | Model | CV Macro F1 | Test Macro F1 |
|---:|---|---|---:|---:|
| 1 | stemming | Linear SVM | 0.789 | 0.801 |
| 2 | raw_tokens | Linear SVM | 0.787 | 0.801 |
| 3 | lemmatization | Linear SVM | 0.785 | 0.803 |

# Best Tuned Model From RandomizedSearchCV

Best parameters:

- `tfidf__sublinear_tf`: `True`
- `tfidf__ngram_range`: `(1, 2)`
- `tfidf__min_df`: `1`
- `tfidf__max_features`: `50000`
- `svm__class_weight`: `balanced`
- `svm__C`: `1.0`

Overall test metrics:

- Accuracy: `0.963`
- Macro F1: `0.801`
- Weighted F1: `0.961`

Full classification report:

```text
              precision    recall  f1-score   support

    negative       0.84      0.76      0.80       485
     neutral       0.70      0.55      0.62       555
    positive       0.98      0.99      0.98     11861

    accuracy                           0.96     12901
   macro avg       0.84      0.77      0.80     12901
weighted avg       0.96      0.96      0.96     12901

```
