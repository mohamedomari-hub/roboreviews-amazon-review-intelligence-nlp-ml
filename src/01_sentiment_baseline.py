"""ML models for sentiment classification.

Run:
    python src/01_sentiment_baseline.py

Goal:
- Use review title + review text as model input.
- Use rating only to create weak labels.
- Compare simple preprocessing + TF-IDF + models.
- Tune the best three baseline setups with RandomizedSearchCV.
"""

from pathlib import Path
import json
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
METRICS = OUT / "metrics"
FIGURES = OUT / "figures"
ERRORS = OUT / "errors"
MODELS = ROOT / "models"

for folder in [METRICS, FIGURES, ERRORS, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all raw Amazon review CSV files."""
    dfs = []
    for path in RAW_DIR.glob("*.csv"):
        df = pd.read_csv(path, low_memory=False)
        df["source_file"] = path.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def make_labels(df):
    """Create text input and rating-derived weak sentiment labels."""
    data = pd.DataFrame()
    data["product_name"] = df.get("name", "")
    data["review_title"] = df.get("reviews.title", "").fillna("")
    data["review_text"] = df.get("reviews.text", "").fillna("")
    data["rating"] = pd.to_numeric(df.get("reviews.rating"), errors="coerce")

    # Rating is not a model input. It is only used to create weak labels.
    data["text"] = (data["review_title"] + " " + data["review_text"]).str.strip()
    data = data.dropna(subset=["rating"])
    data = data[data["text"].str.len() > 0].copy()

    # The raw files overlap, so we remove exact duplicate reviews before
    # splitting. Otherwise the same review could appear in both train and test.
    before = len(data)
    data = data.drop_duplicates(subset=["product_name", "review_title", "review_text", "rating"]).copy()
    print(f"Removed duplicate review rows: {before - len(data)}")

    data["label"] = "positive"
    data.loc[data["rating"] <= 2, "label"] = "negative"
    data.loc[data["rating"] == 3, "label"] = "neutral"
    return data


def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def raw_tokens(text):
    """Keep the review text almost as-is.

    TF-IDF already lowercases text by default and can learn short phrases with
    n-grams, so this baseline does not manually join negations such as
    "not good".
    """
    return str(text)


def stem_text(text):
    text = basic_clean(text)
    try:
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()
        return " ".join(stemmer.stem(word) for word in text.split())
    except Exception:
        return text


def lemmatize_text(text):
    text = basic_clean(text)
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        return " ".join(lemmatizer.lemmatize(word) for word in text.split())
    except Exception:
        return text


def apply_preprocessing(texts, method):
    """Keep preprocessing explicit and easy to read."""
    if method == "basic_clean":
        return texts.apply(basic_clean)
    if method == "stemming":
        return texts.apply(stem_text)
    if method == "lemmatization":
        return texts.apply(lemmatize_text)
    return texts.apply(raw_tokens)


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def evaluate_model(model, vectorizer, X_train, X_test, y_train, y_test, use_numeric_labels=False):
    """Fit vectorizer + model and calculate metrics."""
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if use_numeric_labels:
        y_train_fit = y_train.map(LABEL_TO_ID)
        model.fit(X_train_vec, y_train_fit)
        preds = pd.Series(model.predict(X_test_vec)).map(ID_TO_LABEL)
    else:
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "weighted_f1": f1_score(y_test, preds, average="weighted"),
        "negative_precision": report["negative"]["precision"],
        "negative_recall": report["negative"]["recall"],
        "negative_f1": report["negative"]["f1-score"],
        "neutral_precision": report["neutral"]["precision"],
        "neutral_recall": report["neutral"]["recall"],
        "neutral_f1": report["neutral"]["f1-score"],
        "positive_precision": report["positive"]["precision"],
        "positive_recall": report["positive"]["recall"],
        "positive_f1": report["positive"]["f1-score"],
        "predictions": preds,
    }


def save_readable_model_report(results, tuned_results, tuned_metrics, tuned_report_text):
    """Save a human-readable markdown report with all model performances."""
    report_path = METRICS / "sentiment_model_performance_report.md"

    with open(report_path, "w") as f:
        f.write("# Sentiment Model Performance Report\n\n")
        f.write("Models are sorted by cross-validation macro F1.\n\n")

        for _, row in results.iterrows():
            f.write(f"## {row['preprocessing']} + {row['vectorizer']} + {row['model']}\n\n")
            f.write(f"- CV Macro F1: `{row['cv_macro_f1']:.3f}`\n")
            f.write(f"- Test Accuracy: `{row['accuracy']:.3f}`\n")
            f.write(f"- Test Macro F1: `{row['macro_f1']:.3f}`\n")
            f.write(f"- Test Weighted F1: `{row['weighted_f1']:.3f}`\n\n")
            f.write("| Class | Precision | Recall | F1 |\n")
            f.write("|---|---:|---:|---:|\n")
            f.write(f"| negative | {row['negative_precision']:.3f} | {row['negative_recall']:.3f} | {row['negative_f1']:.3f} |\n")
            f.write(f"| neutral | {row['neutral_precision']:.3f} | {row['neutral_recall']:.3f} | {row['neutral_f1']:.3f} |\n")
            f.write(f"| positive | {row['positive_precision']:.3f} | {row['positive_recall']:.3f} | {row['positive_f1']:.3f} |\n\n")

        f.write("# Tuned Models From RandomizedSearchCV\n\n")
        f.write("The top three fixed baseline setups were tuned. They are sorted by CV Macro F1.\n\n")
        f.write("| Rank | Preprocessing | Model | CV Macro F1 | Test Macro F1 |\n")
        f.write("|---:|---|---|---:|---:|\n")
        for rank, (_, row) in enumerate(tuned_results.iterrows(), start=1):
            f.write(
                f"| {rank} | {row['preprocessing']} | {row['model']} | "
                f"{row['cv_macro_f1']:.3f} | {row['test_macro_f1']:.3f} |\n"
            )

        f.write("\n# Best Tuned Model From RandomizedSearchCV\n\n")
        f.write("Best parameters:\n\n")
        for key, value in tuned_metrics["best_params"].items():
            f.write(f"- `{key}`: `{value}`\n")

        f.write("\nOverall test metrics:\n\n")
        f.write(f"- Accuracy: `{tuned_metrics['test_accuracy']:.3f}`\n")
        f.write(f"- Macro F1: `{tuned_metrics['test_macro_f1']:.3f}`\n")
        f.write(f"- Weighted F1: `{tuned_metrics['test_weighted_f1']:.3f}`\n\n")
        f.write("Full classification report:\n\n")
        f.write("```text\n")
        f.write(tuned_report_text)
        f.write("\n```\n")

    print(f"\nReadable performance report saved: {report_path}")


def save_best_model_report(results, tuned_results, tuned_metrics, tuned_report_text):
    """Save the tuned model and best 3 broad models in the same readable format."""
    report_path = METRICS / "best_tuned_model_classification_report.md"

    with open(report_path, "w") as f:
        f.write("# Sentiment Best Model Report\n\n")
        f.write("## RandomizedSearchCV for tuned TF-IDF + Linear SVM\n\n")
        f.write("Best tuned model\n\n")
        f.write(f"Best preprocessing: {tuned_metrics['preprocessing']}\n")
        f.write(f"Best model: {tuned_metrics['model']}\n\n")
        f.write("Best parameters:\n")
        for key, value in tuned_metrics["best_params"].items():
            f.write(f"  {key}: {value}\n")

        f.write("\nAverage scores for this model:\n")
        f.write(f"  CV Macro F1: {tuned_metrics['cv_macro_f1']:.3f}\n")
        f.write(f"  Test Macro F1: {tuned_metrics['test_macro_f1']:.3f}\n")
        f.write(f"  Weighted F1: {tuned_metrics['test_weighted_f1']:.3f}\n")
        f.write(f"  Accuracy: {tuned_metrics['test_accuracy']:.3f}\n")
        f.write("\nPrecision / Recall / F1 by class:\n\n")
        f.write("```text\n")
        f.write(tuned_report_text)
        f.write("\n```\n\n")

        f.write("## Macro F1 for each setup\n\n")
        f.write("The first rows are the three tuned RandomizedSearchCV setups. The remaining rows are the fixed broad baseline setups.\n\n")
        f.write("| Rank | Setup type | Preprocessing | Vectorizer | Model | Macro F1 |\n")
        f.write("|---:|---|---|---|---|---:|\n")
        rank = 1
        for _, row in tuned_results.iterrows():
            f.write(
                f"| {rank} | Tuned RandomizedSearchCV | {row['preprocessing']} | tfidf | "
                f"{row['model']} | {row['test_macro_f1']:.3f} |\n"
            )
            rank += 1
        for _, row in results.iterrows():
            f.write(
                f"| {rank} | Fixed baseline | {row['preprocessing']} | {row['vectorizer']} | "
                f"{row['model']} | {row['cv_macro_f1']:.3f} |\n"
            )
            rank += 1
        f.write("\n")

        f.write("## Tuned top 3 baseline setups\n\n")
        for rank, (_, row) in enumerate(tuned_results.iterrows(), start=1):
            f.write(f"### {rank}. {row['preprocessing']} + tfidf + {row['model']}\n\n")
            f.write("Average scores for this tuned model:\n")
            f.write(f"  CV Macro F1: {row['cv_macro_f1']:.3f}\n")
            f.write(f"  Test Macro F1: {row['test_macro_f1']:.3f}\n")
            f.write(f"  Weighted F1: {row['test_weighted_f1']:.3f}\n")
            f.write(f"  Accuracy: {row['test_accuracy']:.3f}\n\n")

        f.write("## Best 3 broad baseline models by CV Macro F1\n\n")
        for rank, (_, row) in enumerate(results.head(3).iterrows(), start=1):
            f.write(f"### {rank}. {row['preprocessing']} + {row['vectorizer']} + {row['model']}\n\n")
            f.write("Model setup:\n")
            f.write(f"  Cleaning / preprocessing: {row['preprocessing']}\n")
            f.write(f"  Vectorizer: {row['vectorizer']}\n")
            f.write(f"  Model: {row['model']}\n\n")
            f.write("Average scores for this model:\n")
            f.write(f"  CV Macro F1: {row['cv_macro_f1']:.3f}\n")
            f.write(f"  Test Macro F1: {row['macro_f1']:.3f}\n")
            f.write(f"  Weighted F1: {row['weighted_f1']:.3f}\n")
            f.write(f"  Accuracy: {row['accuracy']:.3f}\n\n")
            f.write("Precision / Recall / F1 by class:\n")
            f.write(f"  Negative: {row['negative_precision']:.3f} / {row['negative_recall']:.3f} / {row['negative_f1']:.3f}\n")
            f.write(f"  Neutral:  {row['neutral_precision']:.3f} / {row['neutral_recall']:.3f} / {row['neutral_f1']:.3f}\n")
            f.write(f"  Positive: {row['positive_precision']:.3f} / {row['positive_recall']:.3f} / {row['positive_f1']:.3f}\n\n")

    print(f"Best model report saved: {report_path}")


def save_macro_f1_bar_charts(results, tuned_results):
    """Save two fair plots: baseline CV comparison and tuned CV/test comparison."""
    top_baselines = results.head(12).copy()
    top_baselines["setup"] = (
        top_baselines["preprocessing"] + " + " + top_baselines["vectorizer"] + "\n" + top_baselines["model"]
    )

    plt.figure(figsize=(10, 7))
    bars = plt.barh(top_baselines["setup"], top_baselines["cv_macro_f1"], color="#64748b")
    plt.gca().invert_yaxis()
    plt.xlim(0.45, 0.83)
    plt.xlabel("Cross-validation Macro F1 before tuning")
    plt.title("Step 1: Fixed Baseline Comparison Before Tuning")

    for bar in bars:
        score = bar.get_width()
        plt.text(score + 0.005, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center")

    plt.tight_layout()
    baseline_path = FIGURES / "sentiment_baseline_cv_macro_f1.png"
    plt.savefig(baseline_path, dpi=180)
    plt.close()
    print(f"Baseline CV Macro F1 chart saved: {baseline_path}")

    tuned = tuned_results.copy()
    tuned["setup"] = tuned["preprocessing"] + " + TF-IDF\n" + tuned["model"]
    y = range(len(tuned))
    height = 0.35

    plt.figure(figsize=(9, 4.5))
    cv_bars = plt.barh(
        [i - height / 2 for i in y],
        tuned["cv_macro_f1"],
        height=height,
        color="#2563eb",
        label="Tuned CV Macro F1",
    )
    test_bars = plt.barh(
        [i + height / 2 for i in y],
        tuned["test_macro_f1"],
        height=height,
        color="#ef4444",
        label="Tuned final test Macro F1",
    )
    plt.gca().invert_yaxis()
    plt.yticks(list(y), tuned["setup"])
    plt.xlim(0.75, 0.82)
    plt.xlabel("Macro F1")
    plt.title("Step 2: Tuned Top 3 Models After RandomizedSearchCV")
    plt.figtext(
        0.5,
        0.01,
        "Both bars are after tuning: blue = tuned CV selection score, red = tuned final test score.",
        ha="center",
        fontsize=9,
    )
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    for bars in [cv_bars, test_bars]:
        for bar in bars:
            score = bar.get_width()
            plt.text(score + 0.001, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center")

    plt.tight_layout(rect=(0, 0.09, 1, 1))
    tuned_path = FIGURES / "sentiment_tuned_top3_cv_vs_test_macro_f1.png"
    plt.savefig(tuned_path, dpi=180)
    plt.close()
    print(f"Tuned CV vs Test Macro F1 chart saved: {tuned_path}")


def main():
    print("Loading data")
    data = make_labels(load_data())
    print("Rows:", len(data))
    print(data["label"].value_counts())

    X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx = train_test_split(
        data["text"],
        data["label"],
        data.index,
        test_size=0.2,
        stratify=data["label"],
        random_state=42,
    )

    preprocessors = ["raw_tokens", "basic_clean", "stemming", "lemmatization"]
    vectorizers = {
        "tfidf": TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=15000, sublinear_tf=True),
    }
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42, max_iter=10000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    rows = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("\nComparing preprocessing, vectorizers, and models")
    for prep_name in preprocessors:
        X_train = apply_preprocessing(X_train_raw, prep_name)
        X_test = apply_preprocessing(X_test_raw, prep_name)

        for vec_name, vectorizer in vectorizers.items():
            for model_name, model in models.items():
                print(f"{prep_name} | {vec_name} | {model_name}")
                use_numeric_labels = model_name == "XGBoost"
                result = evaluate_model(
                    model,
                    vectorizer,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    use_numeric_labels=use_numeric_labels,
                )

                # Simple 3-fold CV on the training set only.
                # XGBoost is much slower on TF-IDF text, so we keep it in the
                # train/test comparison but do not cross-validate it.
                if model_name == "XGBoost":
                    cv_macro_f1 = result["macro_f1"]
                else:
                    pipe = Pipeline([("vectorizer", vectorizer), ("model", model)])
                    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
                    cv_macro_f1 = cv_scores.mean()

                row = {
                    "preprocessing": prep_name,
                    "vectorizer": vec_name,
                    "model": model_name,
                    "cv_macro_f1": cv_macro_f1,
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"],
                    "weighted_f1": result["weighted_f1"],
                    "negative_precision": result["negative_precision"],
                    "negative_recall": result["negative_recall"],
                    "negative_f1": result["negative_f1"],
                    "neutral_precision": result["neutral_precision"],
                    "neutral_recall": result["neutral_recall"],
                    "neutral_f1": result["neutral_f1"],
                    "positive_precision": result["positive_precision"],
                    "positive_recall": result["positive_recall"],
                    "positive_f1": result["positive_f1"],
                }
                rows.append(row)

    results = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    results.to_csv(METRICS / "sentiment_baseline_results.csv", index=False)
    print("\nTop baseline rows:")
    print(results.head())

    print("\nRandomizedSearchCV for the top 3 baseline setups")
    top_3 = results.head(3)
    all_search_rows = []
    tuned_rows = []
    best_search = None
    best_cv_score = -1
    best_preprocessing = None
    best_model_name = None
    best_X_train_for_search = None
    best_X_test_for_search = None

    search_space = {
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "tfidf__min_df": [1, 2, 3],
        "tfidf__max_features": [15000, 30000, 50000],
        "tfidf__sublinear_tf": [True],
        "svm__C": [0.1, 0.5, 1.0, 2.0],
        "svm__class_weight": [None, "balanced"],
    }

    for tune_rank, (_, baseline_row) in enumerate(top_3.iterrows(), start=1):
        prep_name = baseline_row["preprocessing"]
        model_name = baseline_row["model"]
        print(f"Tuning {tune_rank}: {prep_name} + TF-IDF + {model_name}")

        # The top baseline models are Linear SVM setups. Keeping the tuned
        # step focused on SVM makes the comparison easy to explain.
        X_train_for_search = apply_preprocessing(X_train_raw, prep_name)
        X_test_for_search = apply_preprocessing(X_test_raw, prep_name)

        search_pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("svm", LinearSVC(random_state=42, max_iter=10000)),
            ]
        )
        search = RandomizedSearchCV(
            search_pipe,
            search_space,
            n_iter=24,
            cv=3,
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train_for_search, y_train)

        search_results = pd.DataFrame(search.cv_results_)
        search_results["tuned_setup_rank"] = tune_rank
        search_results["preprocessing"] = prep_name
        search_results["model"] = model_name
        all_search_rows.append(search_results)

        preds = search.predict(X_test_for_search)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        tuned_rows.append(
            {
                "preprocessing": prep_name,
                "vectorizer": "tfidf",
                "model": model_name,
                "cv_macro_f1": search.best_score_,
                "test_accuracy": accuracy_score(y_test, preds),
                "test_macro_f1": f1_score(y_test, preds, average="macro"),
                "test_weighted_f1": f1_score(y_test, preds, average="weighted"),
                "negative_f1": report["negative"]["f1-score"],
                "neutral_f1": report["neutral"]["f1-score"],
                "positive_f1": report["positive"]["f1-score"],
                "best_params": search.best_params_,
            }
        )

        if search.best_score_ > best_cv_score:
            best_cv_score = search.best_score_
            best_search = search
            best_preprocessing = prep_name
            best_model_name = model_name
            best_X_train_for_search = X_train_for_search
            best_X_test_for_search = X_test_for_search

    pd.concat(all_search_rows, ignore_index=True).to_csv(METRICS / "random_search_sentiment_results.csv", index=False)
    tuned_results = pd.DataFrame(tuned_rows).sort_values("cv_macro_f1", ascending=False)
    tuned_results.to_csv(METRICS / "top3_tuned_sentiment_results.csv", index=False)

    search = best_search
    X_train_for_search = best_X_train_for_search
    X_test_for_search = best_X_test_for_search
    tuned_preds = search.predict(X_test_for_search)
    tuned_report_text = classification_report(y_test, tuned_preds, zero_division=0)
    tuned_report = classification_report(y_test, tuned_preds, output_dict=True, zero_division=0)
    tuned_metrics = {
        "preprocessing": best_preprocessing,
        "model": best_model_name,
        "cv_macro_f1": best_cv_score,
        "best_params": search.best_params_,
        "test_accuracy": accuracy_score(y_test, tuned_preds),
        "test_macro_f1": f1_score(y_test, tuned_preds, average="macro"),
        "test_weighted_f1": f1_score(y_test, tuned_preds, average="weighted"),
        "negative_f1": tuned_report["negative"]["f1-score"],
        "neutral_f1": tuned_report["neutral"]["f1-score"],
        "positive_f1": tuned_report["positive"]["f1-score"],
    }

    print("\nBest tuned model")
    print(f"Best preprocessing: {tuned_metrics['preprocessing']}")
    print(f"Best model: {tuned_metrics['model']}")
    print("Best parameters:")
    for key, value in search.best_params_.items():
        print(f"  {key}: {value}")
    print("\nOptimal model test metrics:")
    print(f"  CV Macro F1: {tuned_metrics['cv_macro_f1']:.3f}")
    print(f"  Accuracy:    {tuned_metrics['test_accuracy']:.3f}")
    print(f"  Macro F1:    {tuned_metrics['test_macro_f1']:.3f}")
    print(f"  Weighted F1: {tuned_metrics['test_weighted_f1']:.3f}")
    print("\nPrecision / Recall / F1 by class:")
    print(tuned_report_text)

    with open(METRICS / "best_sentiment_baseline_metrics.json", "w") as f:
        json.dump(tuned_metrics, f, indent=2)

    save_best_model_report(results, tuned_results, tuned_metrics, tuned_report_text)
    save_readable_model_report(results, tuned_results, tuned_metrics, tuned_report_text)
    save_macro_f1_bar_charts(results, tuned_results)

    joblib.dump(search.best_estimator_, MODELS / "best_sentiment_baseline_model.joblib")

    matrix = confusion_matrix(y_test, tuned_preds, labels=["negative", "neutral", "positive"])
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "neutral", "positive"], yticklabels=["negative", "neutral", "positive"])
    plt.title("Best Sentiment Baseline Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Rating label")
    plt.tight_layout()
    plt.savefig(FIGURES / "best_sentiment_baseline_confusion_matrix.png", dpi=160)
    plt.close()

    row_totals = matrix.sum(axis=1, keepdims=True)
    matrix_percent = matrix / row_totals * 100
    labels = []
    for i in range(matrix.shape[0]):
        row = []
        for j in range(matrix.shape[1]):
            row.append(f"{matrix[i, j]:,}\n{matrix_percent[i, j]:.1f}%")
        labels.append(row)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix_percent,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["Predicted negative", "Predicted neutral", "Predicted positive"],
        yticklabels=["Actual negative", "Actual neutral", "Actual positive"],
        cbar_kws={"label": "Percent of actual class"},
    )
    plt.title("Best Tuned Sentiment Model: Confusion Matrix")
    plt.xlabel("Model prediction")
    plt.ylabel("Rating-derived label")
    plt.tight_layout()
    plt.savefig(FIGURES / "best_tuned_sentiment_confusion_matrix_readable.png", dpi=180)
    plt.close()

    print("\nDownsampling experiment")
    train_small = pd.DataFrame({"text": X_train_for_search, "label": y_train})
    min_size = train_small["label"].value_counts().min()
    downsampled = train_small.groupby("label", group_keys=False).apply(lambda x: x.sample(min_size, random_state=42))
    search.best_estimator_.fit(downsampled["text"], downsampled["label"])
    down_preds = search.best_estimator_.predict(X_test_for_search)

    print("\nRating-text discrepancy filtering experiment")
    search.best_estimator_.fit(X_train_for_search, y_train)
    train_preds = search.best_estimator_.predict(X_train_for_search)
    keep = train_preds == y_train.values
    search.best_estimator_.fit(X_train_for_search[keep], y_train[keep])
    filtered_preds = search.best_estimator_.predict(X_test_for_search)

    extra = {
        "downsampling_macro_f1": f1_score(y_test, down_preds, average="macro"),
        "filtered_label_macro_f1": f1_score(y_test, filtered_preds, average="macro"),
        "original_tuned_macro_f1": tuned_metrics["test_macro_f1"],
        "note": "Downsampling and filtering are compared against the tuned model from this script.",
    }
    with open(METRICS / "sentiment_extra_experiments.json", "w") as f:
        json.dump(extra, f, indent=2)

    print("\nSaving contradiction examples")
    test_rows = data.loc[test_idx].copy()
    test_rows["predicted_sentiment"] = tuned_preds
    test_rows["true_label_from_rating"] = test_rows["label"]
    test_rows["contradiction_type"] = ""
    test_rows.loc[(test_rows["rating"] <= 2) & (test_rows["predicted_sentiment"] == "positive"), "contradiction_type"] = "low_rating_positive_text"
    test_rows.loc[(test_rows["rating"] >= 4) & (test_rows["predicted_sentiment"] == "negative"), "contradiction_type"] = "high_rating_negative_text"
    test_rows.loc[(test_rows["rating"] == 3) & (test_rows["predicted_sentiment"] == "positive"), "contradiction_type"] = "neutral_rating_positive_text"
    test_rows.loc[(test_rows["rating"] == 3) & (test_rows["predicted_sentiment"] == "negative"), "contradiction_type"] = "neutral_rating_negative_text"
    contradictions = test_rows[test_rows["contradiction_type"] != ""]
    cols = ["product_name", "review_title", "review_text", "rating", "true_label_from_rating", "predicted_sentiment", "contradiction_type"]
    contradictions[cols].to_csv(ERRORS / "rating_text_contradictions.csv", index=False)

    with open(ERRORS / "rating_text_contradictions_examples.md", "w") as f:
        f.write("# Rating-text contradiction examples\n\n")
        f.write("Ratings are weak labels. Some review text does not fully match the star rating.\n\n")
        for kind, group in contradictions.groupby("contradiction_type"):
            f.write(f"## {kind}\n\n")
            for _, row in group.head(3).iterrows():
                f.write(f"- Rating {row['rating']}, predicted {row['predicted_sentiment']}: {row['review_title']}\n")

    print("\nDone. Sentiment outputs saved in outputs/metrics, outputs/errors, outputs/figures, and models.")


if __name__ == "__main__":
    main()
