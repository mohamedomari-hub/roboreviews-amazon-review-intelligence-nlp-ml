"""Simple clustering baseline.

Run:
    python src/02_clustering_baseline.py

Why this setup:
- Sentiment experiments showed stemming, raw text, and lemmatization were close.
- For clustering, lemmatization is easier to explain because words stay readable.
- We use only lemmatization + TF-IDF to keep the clustering code simple.

The script compares two scenarios:
1. before_with_reviews: product name + category + review title + review text
2. after_canonical_no_reviews: product-family tokens inferred from the product name only

Then it compares:
- KMeans
- Hierarchical clustering
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
METRICS = OUT / "metrics"
FIGURES = OUT / "figures"
CLUSTERING = OUT / "clustering"

for folder in [METRICS, FIGURES, CLUSTERING]:
    folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. Simple text cleaning
# ------------------------------------------------------------


def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9_\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lemmatize_text(text):
    """Basic cleaning first, then lemmatization."""
    text = basic_clean(text)
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        return " ".join(lemmatizer.lemmatize(word) for word in text.split())
    except Exception:
        return text


def repeat_text(text, times):
    """Repeat important tokens so TF-IDF gives them more weight."""
    text = str(text).strip()
    if not text or text.lower() == "nan":
        return ""
    return " ".join([text] * times)


def has_any(text, terms):
    """Check if any word/phrase from a list is inside a text."""
    return any(term in text for term in terms)


def clean_product_name_for_grouping(name):
    """Fix product names that contain duplicated or pasted-together names."""
    if pd.isna(name):
        return ""

    text = str(name).replace("\\r", "\n").replace("\\n", "\n")
    parts = re.split(r",,,|\n|\"", text)
    parts = [part.strip(" ,") for part in parts if part.strip(" ,")]

    if not parts:
        return str(name).strip()

    # In this dataset the first part is usually the real product name.
    # Example: "Echo (White),,, Fire Tablet..." should stay "Echo (White)".
    best_part = parts[0]

    # Remove repeated "Amazon - Amazon" noise after choosing the best part.
    best_part = re.sub(r"\bAmazon\s*-\s*Amazon\b", "Amazon", best_part, flags=re.IGNORECASE)
    best_part = re.sub(r"\s+", " ", best_part).strip(" ,")
    return best_part


# ------------------------------------------------------------
# 2. Product text preparation
# ------------------------------------------------------------


def canonicalize_product_name(name):
    """Remove product variants so similar products have closer names."""
    text = basic_clean(name)

    # Remove repeated marketplace prefix noise.
    text = re.sub(r"\bamazon\s+amazon\b", "amazon", text)

    phrases_to_remove = [
        "includes special offers",
        "with special offers",
        "without special offers",
        "special offers",
        "certified refurbished",
        "previous generation",
        "brand new",
        "high resolution",
        "touchscreen",
        "display",
        "wifi",
        "wi fi",
        "alexa",
        "amazon",
        "kindle",
        "fire",
        "case",
        "cover",
        "kid proof",
        "kids edition",
        "edition",
    ]
    for phrase in phrases_to_remove:
        text = text.replace(phrase, " ")

    text = re.sub(r"\b\d+\s*gb\b", " ", text)
    text = re.sub(r"\b\d+\s*ppi\b", " ", text)
    text = re.sub(r"\b(black|white|blue|red|green|magenta|tangerine|silver|merlot|yellow)\b", " ", text)
    text = re.sub(r"\b(6|7|8|10)\s*(inch|inches|in)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_product_family(product_name, category):
    """Create one clear product-family token from the product name."""
    name = str(product_name).lower()

    # Order matters: very specific patterns should be checked first.
    if has_any(name, ["battery", "batteries", "alkaline"]):
        return "family_accessory_battery"

    if has_any(name, ["power adapter", "charger", "charging cable", "powerfast", "adapter", "cable", "speaker wire"]):
        return "family_accessory_power_charger"

    if has_any(name, ["kids edition tablet", "kid-proof case"]):
        return "family_fire_tablet"

    if has_any(name, ["sleeve", "case", "cover", "screen protector", "protector", "stand", "bag", "keyboard"]):
        return "family_accessory_case_cover"

    if has_any(name, ["fire tablet", "fire hd", "kindle fire", "fire 7", "fire 16gb"]):
        return "family_fire_tablet"

    if has_any(name, ["fire tv", "fire stick", "streaming media", "streaming player"]):
        return "family_fire_tv_streaming"

    if has_any(name, ["paperwhite", "voyage", "oasis", "e-reader", "ereader"]):
        return "family_kindle_ereader"

    if "kindle" in name and "fire" not in name:
        return "family_kindle_ereader"

    if has_any(name, ["echo", "alexa", "amazon tap", "bluetooth speaker", "speaker"]):
        return "family_echo_alexa_speaker"

    if has_any(
        name,
        [
            "file folder",
            "document organizer",
            "backpack",
            "binder",
            "pet",
            "kennel",
            "litter",
            "cat",
            "dog",
            "hot handle",
            "coconut water",
            "red tea",
            "nespresso pod storage drawer",
        ],
    ):
        return "family_non_electronic"

    if "amazonbasics" in name:
        return "family_accessory_general"

    return "family_other_amazon_product"


def infer_structured_tokens(product_name, category):
    """Add helper tokens only when those words appear in the product name."""
    name = str(product_name).lower()
    family = infer_product_family(product_name, category)
    tokens = [family]

    # These helper tokens are still manual labels, but each one is triggered
    # only by a word that exists in the product name.
    word_to_token = {
        "tablet": "word_tablet",
        "touchscreen": "word_touchscreen",
        "display": "word_display",
        "screen": "word_screen",
        "kids": "word_kids",
        "kid": "word_kids",
        "e-reader": "word_ereader",
        "ereader": "word_ereader",
        "paperwhite": "word_paperwhite",
        "voyage": "word_voyage",
        "oasis": "word_oasis",
        "echo": "word_echo",
        "alexa": "word_alexa",
        "speaker": "word_speaker",
        "tap": "word_tap",
        "fire tv": "word_fire_tv",
        "fire stick": "word_fire_stick",
        "streaming": "word_streaming",
        "remote": "word_remote",
        "battery": "word_battery",
        "batteries": "word_battery",
        "alkaline": "word_alkaline",
        "aa": "word_aa",
        "aaa": "word_aaa",
        "charger": "word_charger",
        "adapter": "word_adapter",
        "power": "word_power",
        "cable": "word_cable",
        "case": "word_case",
        "cover": "word_cover",
        "sleeve": "word_sleeve",
        "bag": "word_bag",
        "keyboard": "word_keyboard",
        "stand": "word_stand",
        "protector": "word_protector",
        "pet": "word_pet",
        "dog": "word_dog",
        "cat": "word_cat",
        "kennel": "word_kennel",
        "litter": "word_litter",
        "folder": "word_folder",
        "document": "word_document",
        "nespresso": "word_nespresso",
    }

    for word, token in word_to_token.items():
        if word in name and token not in tokens:
            tokens.append(token)

    return " ".join(tokens)


def load_product_table():
    """Load raw reviews and aggregate them to one row per product."""
    rows = []

    # Step 1: read all CSV review rows.
    for path in RAW_DIR.glob("*.csv"):
        df = pd.read_csv(path, low_memory=False)
        for _, row in df.iterrows():
            original_product_name = row.get("name", "")
            product_name = clean_product_name_for_grouping(original_product_name)
            category = row.get("categories", row.get("primaryCategories", ""))
            review_title = row.get("reviews.title", "")
            review_text = row.get("reviews.text", "")
            rows.append(
                {
                    "product_name": product_name,
                    "original_product_name": original_product_name,
                    "category": category,
                    "review_title": review_title,
                    "review_text": review_text,
                    "before_text": f"{product_name} {category} {review_title} {review_text}",
                }
            )

    # Step 2: remove empty names and exact duplicated review rows.
    reviews = pd.DataFrame(rows)
    reviews = reviews[reviews["product_name"].astype(str).str.len() > 0].copy()
    reviews = reviews.drop_duplicates(subset=["product_name", "category", "review_title", "review_text"])

    # Step 3: aggregate reviews to one row per product.
    products = (
        reviews.groupby("product_name")
        .agg(
            category=("category", "first"),
            original_product_name=("original_product_name", "first"),
            review_rows=("before_text", "size"),
            before_text=("before_text", lambda values: " ".join(values.astype(str))),
        )
        .reset_index()
    )

    # Step 4: create simple product identity features.
    products["canonical_product_name"] = products["product_name"].apply(canonicalize_product_name)
    products["product_family"] = products.apply(
        lambda row: infer_product_family(row["product_name"], row["category"]),
        axis=1,
    )
    products["structured_tokens"] = products.apply(
        lambda row: infer_structured_tokens(row["product_name"], row["category"]),
        axis=1,
    )

    # This is the improved second-scenario representation.
    # It intentionally removes reviews, category text, and noisy product variants.
    # Each row becomes a clean product-identity string made from the name only.
    products["after_text"] = products.apply(
        lambda row: " ".join(
            [
                repeat_text(row["product_family"], 100),
                repeat_text(row["structured_tokens"], 20),
            ]
        ),
        axis=1,
    )
    return products


# ------------------------------------------------------------
# 3. Vectorization
# ------------------------------------------------------------


def make_vectors(texts):
    """Lemmatize text and convert it to TF-IDF vectors."""
    cleaned_text = texts.apply(lemmatize_text)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=8000,
        max_df=0.90,
        sublinear_tf=True,
        stop_words="english",
    )
    X = vectorizer.fit_transform(cleaned_text)
    return X, vectorizer


def reduce_vectors_for_clustering(X):
    """Make sparse TF-IDF easier to cluster and compare."""
    # A small SVD keeps the main product-family signals and removes tiny noise.
    # This makes the canonical product clusters easier to separate.
    n_components = min(8, X.shape[0] - 1, X.shape[1] - 1)
    if n_components >= 2:
        reduced = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(X)
    else:
        reduced = X.toarray()
    return normalize(reduced)


# ------------------------------------------------------------
# 4. Clustering scans
# ------------------------------------------------------------


def elbow_score(cluster_features, labels):
    """Simple elbow-style score that works for KMeans and HC.

    KMeans has a built-in inertia score. Hierarchical clustering does not.
    To compare all methods in one plot, we calculate the same simple idea:
    how far each product is from the center of its assigned cluster.
    Lower values are better.
    """
    total_distance = 0

    for cluster_id in sorted(set(labels)):
        points = cluster_features[labels == cluster_id]
        center = points.mean(axis=0)
        total_distance += ((points - center) ** 2).sum()

    return float(total_distance)


def scan_kmeans(cluster_features):
    rows = []
    for k in range(2, 21):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(cluster_features)
        rows.append(
            {
                "method": "KMeans",
                "k": k,
                "inertia": model.inertia_,
                "elbow_score": elbow_score(cluster_features, labels),
                "silhouette_score": silhouette_score(cluster_features, labels, metric="cosine"),
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans(cluster_features, k):
    """Fit one KMeans model."""
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    return model.fit_predict(cluster_features)


def scan_hierarchical(cluster_features):
    rows = []
    for k in range(2, 21):
        try:
            model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        except TypeError:
            model = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")

        labels = model.fit_predict(cluster_features)
        rows.append(
            {
                "method": "Hierarchical",
                "k": k,
                "inertia": "",
                "elbow_score": elbow_score(cluster_features, labels),
                "silhouette_score": silhouette_score(cluster_features, labels, metric="cosine"),
            }
        )
    return pd.DataFrame(rows)


def fit_hierarchical(cluster_features, k):
    """Fit one Hierarchical clustering model with cosine distance."""
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")

    return model.fit_predict(cluster_features)


# ------------------------------------------------------------
# 5. Plots and readable outputs
# ------------------------------------------------------------


def plot_method_comparison(kmeans_df, hierarchical_df, scenario_name):
    """Plot silhouette and elbow-style scores for all available methods."""
    fig, left_axis = plt.subplots(figsize=(10, 5.5))

    left_axis.plot(kmeans_df["k"], kmeans_df["silhouette_score"], marker="o", label="KMeans silhouette")
    left_axis.plot(hierarchical_df["k"], hierarchical_df["silhouette_score"], marker="s", label="Hierarchical silhouette")

    left_axis.set_xlabel("Number of clusters (k)")
    left_axis.set_ylabel("Silhouette score")

    right_axis = left_axis.twinx()
    right_axis.plot(
        kmeans_df["k"],
        kmeans_df["elbow_score"],
        marker="D",
        color="#64748b",
        alpha=0.55,
        label="KMeans elbow score",
    )
    right_axis.plot(
        hierarchical_df["k"],
        hierarchical_df["elbow_score"],
        marker="x",
        linestyle=":",
        color="#94a3b8",
        alpha=0.85,
        label="Hierarchical elbow score",
    )
    right_axis.set_ylabel("Elbow score: within-cluster distance")

    lines_left, labels_left = left_axis.get_legend_handles_labels()
    lines_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(lines_left + lines_right, labels_left + labels_right, loc="best")

    plt.title(f"Silhouette + Elbow by Method ({scenario_name})")
    fig.tight_layout()
    plt.savefig(FIGURES / f"{scenario_name}_method_silhouette_and_elbow.png", dpi=160)
    plt.close()


def get_pca_display_groups(labels, products, scenario_name):
    """Choose the names/colors shown in the PCA plot.

    Before taxonomy correction, the plot shows KMeans cluster numbers.
    After taxonomy correction, the plot shows readable product categories.
    """
    if products is None or scenario_name != "after_canonical_no_reviews":
        return pd.Series([f"Cluster {cluster_id}" for cluster_id in labels])

    named_products = products.copy()
    named_products["cluster_id"] = labels
    named_products[["corrected_category", "correction_reason"]] = named_products.apply(taxonomy_rule, axis=1)
    return named_products["corrected_category"]


def get_readable_cluster_names(labels, products, scenario_name):
    """Name each cluster using its dominant taxonomy category."""
    cluster_ids = sorted(set(labels))

    if products is None or scenario_name != "after_canonical_no_reviews":
        return {cluster_id: f"Cluster {cluster_id}" for cluster_id in cluster_ids}

    named_products = products.copy()
    named_products["cluster_id"] = labels
    named_products[["corrected_category", "correction_reason"]] = named_products.apply(taxonomy_rule, axis=1)

    dominant_categories = {}
    for cluster_id, group in named_products.groupby("cluster_id"):
        dominant_categories[cluster_id] = group["corrected_category"].value_counts().idxmax()

    # If one category appears in more than one cluster, add A/B/C so we can
    # still see all k clusters in the plot.
    category_counts = pd.Series(dominant_categories).value_counts()
    category_seen = {}
    cluster_names = {}

    for cluster_id in cluster_ids:
        category = dominant_categories[cluster_id]
        if category_counts[category] > 1:
            number_seen = category_seen.get(category, 0)
            suffix = chr(ord("A") + number_seen)
            category_seen[category] = number_seen + 1
            cluster_names[cluster_id] = f"{category} {suffix}"
        else:
            cluster_names[cluster_id] = category

    return cluster_names


def draw_cluster_ellipse(axis, x_values, y_values, color):
    """Draw a simple oval around one cluster in a PCA panel."""
    if len(x_values) == 0:
        return

    # Use percentiles instead of min/max so one far point does not create a
    # giant ellipse that covers the whole figure.
    x_min, x_max = pd.Series(x_values).quantile([0.10, 0.90]).tolist()
    y_min, y_max = pd.Series(y_values).quantile([0.10, 0.90]).tolist()
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Add padding so the oval does not sit directly on the points.
    width = min(max((x_max - x_min) * 1.45, 0.06), 0.55)
    height = min(max((y_max - y_min) * 1.45, 0.06), 0.55)

    ellipse = Ellipse(
        (x_center, y_center),
        width=width,
        height=height,
        facecolor=color,
        edgecolor=color,
        alpha=0.08,
        linewidth=2,
        zorder=1,
    )
    axis.add_patch(ellipse)


def draw_original_cluster_ellipses(axis, coords, labels, x_id, y_id, cluster_names=None, show_labels=False):
    """Draw compact ellipses around the actual clusters."""
    ellipse_palette = sns.color_palette("Set2", n_colors=len(sorted(set(labels))))

    for color, cluster_id in zip(ellipse_palette, sorted(set(labels))):
        mask = labels == cluster_id
        draw_cluster_ellipse(axis, coords[mask, x_id], coords[mask, y_id], color)
        if show_labels and cluster_names is not None:
            x_center = pd.Series(coords[mask, x_id]).median()
            y_center = pd.Series(coords[mask, y_id]).median()
            axis.text(
                x_center,
                y_center,
                cluster_names[cluster_id],
                fontsize=8,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.82},
                zorder=3,
            )


def keep_unique_legend_items(handles, legend_labels):
    """Remove duplicate category names from a legend."""
    unique_handles = []
    unique_labels = []

    for handle, label in zip(handles, legend_labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)

    return unique_handles, unique_labels


def save_pca_plot(X, labels, scenario_name, products=None, method_name="KMeans", output_tag=None, color_mode="taxonomy"):
    if output_tag is None:
        output_tag = scenario_name

    cluster_features = reduce_vectors_for_clustering(X)
    n_pca_components = min(5, cluster_features.shape[1])
    pca = PCA(n_components=n_pca_components, random_state=42)
    coords = pca.fit_transform(cluster_features)
    cluster_names = get_readable_cluster_names(labels, products, scenario_name)

    if color_mode == "clusters":
        display_groups = pd.Series([cluster_names[cluster_id] for cluster_id in labels])
        legend_title = f"{method_name} cluster"
        color_note = "point colors and legend show readable cluster names"
    else:
        display_groups = get_pca_display_groups(labels, products, scenario_name)
        legend_title = "Cluster group"
        color_note = "point colors show readable taxonomy categories after correction"

    number_of_clusters = len(set(labels))
    group_names = sorted(display_groups.unique())
    palette = sns.color_palette("tab10", n_colors=len(group_names))
    color_by_group = dict(zip(group_names, palette))

    pc1_variance = pca.explained_variance_ratio_[0] * 100
    pc2_variance = pca.explained_variance_ratio_[1] * 100
    total_variance = pc1_variance + pc2_variance
    readable_name = scenario_name.replace("_", " ").title()

    plt.figure(figsize=(8, 6))
    axis = plt.gca()
    draw_original_cluster_ellipses(axis, coords, labels, 0, 1, cluster_names, show_labels=False)
    for group_name in group_names:
        mask = display_groups.values == group_name
        color = color_by_group[group_name]
        axis.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=group_name,
            color=color,
            s=45,
            zorder=2,
        )
    handles, legend_labels = axis.get_legend_handles_labels()
    handles, legend_labels = keep_unique_legend_items(handles, legend_labels)
    plt.xlabel(f"PC1 ({pc1_variance:.1f}% variance)")
    plt.ylabel(f"PC2 ({pc2_variance:.1f}% variance)")
    plt.title(f"PCA View of Selected {method_name} Clusters ({scenario_name})\n2D view shows {total_variance:.1f}% of variance")
    if scenario_name == "after_canonical_no_reviews":
        plt.figtext(
            0.5,
            0.01,
            f"Ellipses show the {number_of_clusters} {method_name} clusters; {color_note}.",
            ha="center",
            fontsize=9,
        )
    plt.legend(handles, legend_labels, title=legend_title, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES / f"{output_tag}_selected_clustering_pca_2d.png", dpi=160)
    plt.close()

    # Sometimes PC1 vs PC2 is not the clearest view.
    # This second figure shows many PCA pairs so we can pick the best one.
    pca_pairs = []
    for first_component in range(n_pca_components):
        for second_component in range(first_component + 1, n_pca_components):
            first_variance = pca.explained_variance_ratio_[first_component] * 100
            second_variance = pca.explained_variance_ratio_[second_component] * 100
            pca_pairs.append(
                (
                    first_component,
                    second_component,
                    f"PC{first_component + 1} ({first_variance:.1f}%)",
                    f"PC{second_component + 1} ({second_variance:.1f}%)",
                )
            )

    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    axes = axes.flatten()

    legend_handles = []
    legend_labels = []

    for axis, (x_id, y_id, x_label, y_label) in zip(axes, pca_pairs):
        draw_original_cluster_ellipses(axis, coords, labels, x_id, y_id)
        for group_name in group_names:
            mask = display_groups.values == group_name
            color = color_by_group[group_name]
            points = axis.scatter(
                coords[mask, x_id],
                coords[mask, y_id],
                color=color,
                s=30,
                label=group_name,
                zorder=2,
            )
            if group_name not in legend_labels:
                legend_handles.append(points)
                legend_labels.append(group_name)

        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(f"{x_label.split()[0]} vs {y_label.split()[0]}")

    for empty_axis in axes[len(pca_pairs):]:
        empty_axis.axis("off")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title=legend_title,
            loc="center right",
            bbox_to_anchor=(0.99, 0.52),
            ncol=1,
            frameon=True,
        )

    fig.suptitle(f"PCA Cluster Views - {readable_name}", fontsize=16, y=0.98)
    fig.text(
        0.5,
        0.04,
        f"Each panel shows a different pair of PCA axes. Ellipses show {number_of_clusters} {method_name} clusters; {color_note}. PCA is only for visualization.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.08, 0.93, 0.92])
    plt.savefig(FIGURES / f"{output_tag}_selected_clustering_pca_pairs.png", dpi=160)
    plt.close()


def save_cluster_summary(products, X, vectorizer, labels, scenario_name, method_name="KMeans", output_tag=None):
    if output_tag is None:
        output_tag = scenario_name

    output = products.copy()
    output["cluster_id"] = labels
    output.to_csv(CLUSTERING / f"{output_tag}_selected_product_clusters.csv", index=False)
    cluster_names = get_readable_cluster_names(labels, products, scenario_name)

    cluster_sizes = output["cluster_id"].value_counts().sort_index()
    x_labels = [f"{cluster_id}\n{cluster_names[cluster_id]}" for cluster_id in cluster_sizes.index]
    x_positions = range(len(cluster_sizes.index))

    plt.figure(figsize=(11, 5.5))
    bars = plt.bar(x_positions, cluster_sizes.values, color="#2563eb")
    plt.title(f"Selected {method_name} Cluster Sizes ({scenario_name})")
    plt.xlabel(f"{method_name} cluster ID and dominant group name")
    plt.ylabel("Number of products")
    plt.xticks(x_positions, x_labels, rotation=25, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, int(height), ha="center")

    plt.tight_layout()
    plt.savefig(FIGURES / f"{output_tag}_cluster_sizes.png", dpi=160)
    plt.close()

    feature_names = vectorizer.get_feature_names_out()
    summary_rows = []

    for cluster_id in sorted(output["cluster_id"].unique()):
        cluster_products = output[output["cluster_id"] == cluster_id]
        cluster_mask = (output["cluster_id"].values == cluster_id)
        cluster_matrix = X[cluster_mask]
        top_ids = cluster_matrix.mean(axis=0).A1.argsort()[::-1][:10]
        keywords = [feature_names[i] for i in top_ids]

        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_name": cluster_names[cluster_id],
                "cluster_size": len(cluster_products),
                "example_products": " | ".join(cluster_products["product_name"].head(5)),
                "top_keywords": ", ".join(keywords),
                "notes": "cluster_size means number of products, not review rows",
            }
        )

    pd.DataFrame(summary_rows).to_csv(CLUSTERING / f"{output_tag}_cluster_summary.csv", index=False)


# ------------------------------------------------------------
# 6. Simple taxonomy correction after clustering
# ------------------------------------------------------------


def taxonomy_rule(row):
    """Correct obvious device/accessory conflicts after clustering."""
    name_text = str(row["product_name"]).lower()
    category_text = str(row.get("category", "")).lower()
    text = f"{name_text} {category_text}"

    # These are simple product-name rules. They make the final group names
    # easier to explain than raw cluster numbers.
    non_electronic_terms = [
        "file folder", "document organizer", "pet", "kennel", "litter",
        "cat", "dog", "hot handle", "coconut water", "red tea",
        "nespresso pod storage drawer", "backpack", "binder",
    ]
    strong_accessory_terms = [
        "charger", "charging", "powerfast", "adapter", "cable", "case", "cover",
        "sleeve", "stand", "dock", "keyboard", "screen protector", "protector",
        "replacement", "mount",
    ]

    # Fire TV is a strong product-family signal. Check it early because some
    # raw categories are noisy and mention tablets, Kindle, or office products.
    if has_any(name_text, ["fire tv", "fire stick", "streaming media", "streaming player"]):
        return pd.Series(["Fire TV & Streaming", "matched Fire TV/streaming product name"])

    if has_any(text, non_electronic_terms):
        return pd.Series(["Non-Electronics", "matched non-electronic product terms"])

    # Accessories first, because many accessories contain device words like
    # Echo, Fire TV, Kindle, speaker, case, or cover.
    if has_any(text, ["battery", "batteries", "alkaline", " aa ", " aaa "]):
        return pd.Series(["Electronic Accessories", "matched battery/accessory terms"])

    if has_any(text, ["power adapter", "charger", "charging cable", "powerfast", "speaker wire"]):
        return pd.Series(["Electronic Accessories", "matched power/wire accessory terms"])

    # Fire tablets before Echo/Alexa and broad case/cover rules, because many
    # Fire tablets include Alexa or a kid-proof case in the title.
    if has_any(text, ["fire tablet", "fire hd", "kindle fire", "fire 7", "fire 16gb", "kids edition tablet", "kid-proof case"]):
        return pd.Series(["Fire Tablets", "matched Fire tablet terms"])

    if has_any(text, strong_accessory_terms):
        return pd.Series(["Electronic Accessories", "accessory term overrides device-family terms"])

    if has_any(text, ["fire tv", "fire stick", "streaming media", "streaming player"]):
        return pd.Series(["Fire TV & Streaming", "matched Fire TV/streaming terms"])

    # Check Echo/Alexa speakers before broad Kindle checks. Some Echo/Alexa
    # products have noisy raw category text such as "Kindle Store".
    if has_any(name_text, ["amazon tap", "echo", "alexa", "bluetooth speaker", "speaker"]):
        return pd.Series(["Echo & Alexa Devices", "matched Echo/Alexa/speaker product name"])

    if has_any(text, ["oasis", "paperwhite", "voyage", "e-reader", "ereader"]):
        return pd.Series(["Kindle E-Readers", "matched Kindle/e-reader terms"])

    if "kindle" in name_text and "fire" not in name_text:
        return pd.Series(["Kindle E-Readers", "matched Kindle/e-reader terms"])

    return pd.Series([f"Cluster {row['cluster_id']}", "no taxonomy rule matched"])


def apply_taxonomy_correction():
    path = CLUSTERING / "after_canonical_no_reviews_selected_product_clusters.csv"
    clusters = pd.read_csv(path)
    clusters["original_cluster_id"] = clusters["cluster_id"]
    clusters[["corrected_category", "correction_reason"]] = clusters.apply(taxonomy_rule, axis=1)
    clusters.to_csv(CLUSTERING / "after_canonical_no_reviews_taxonomy_corrected_clusters.csv", index=False)

    corrections = clusters[
        clusters["corrected_category"] != "Cluster " + clusters["original_cluster_id"].astype(str)
    ].copy()
    corrections[["product_name", "original_cluster_id", "corrected_category", "correction_reason"]].to_csv(
        CLUSTERING / "taxonomy_corrections.csv",
        index=False,
    )

    print("\nTaxonomy corrected category counts:")
    print(clusters["corrected_category"].value_counts())


def add_product_group_section(lines, title, csv_name, group_column):
    """Add one product-list section to the markdown report."""
    path = CLUSTERING / csv_name
    if not path.exists():
        lines.append(f"## {title}\n")
        lines.append(f"Missing file: `{csv_name}`\n")
        return

    df = pd.read_csv(path)
    lines.append(f"## {title}\n")
    lines.append(f"Source file: `{csv_name}`\n")
    lines.append(f"Total products: **{len(df)}**\n")

    for group_value, group_df in df.groupby(group_column, sort=True):
        group_df = group_df.sort_values(["review_rows", "product_name"], ascending=[False, True])
        lines.append(f"### {group_column}: {group_value} ({len(group_df)} products)\n")

        for _, row in group_df.iterrows():
            family = row.get("product_family", "")
            family_text = f" | family: {family}" if isinstance(family, str) and family else ""
            lines.append(f"- {row['product_name']} ({row['review_rows']} review rows{family_text})")

        lines.append("")


def save_product_groups_markdown():
    """Save a readable list of products inside each cluster/group."""
    lines = [
        "# Product Lists by Cluster / Group\n",
        "This file lists the products inside each selected clustering result. "
        "`review_rows` means how many review rows were available for that product.\n",
    ]

    add_product_group_section(
        lines,
        "Scenario 1: Before Improvement - With Review Text, Selected KMeans Clusters",
        "before_with_reviews_selected_product_clusters.csv",
        "cluster_id",
    )
    add_product_group_section(
        lines,
        "Scenario 2: After Improvement - Canonical Product Tokens, Selected KMeans Clusters",
        "after_canonical_no_reviews_selected_product_clusters.csv",
        "cluster_id",
    )
    add_product_group_section(
        lines,
        "Scenario 2: Taxonomy-Corrected Product Groups",
        "after_canonical_no_reviews_taxonomy_corrected_clusters.csv",
        "corrected_category",
    )

    output_path = CLUSTERING / "product_groups.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved product group list: {output_path}")
    print("Preview:")
    print("\n".join(lines[:35]))


# ------------------------------------------------------------
# 7. Run one complete clustering scenario
# ------------------------------------------------------------


def run_scenario(products, text_column, scenario_name):
    print(f"\n=== {scenario_name} ===")
    print("Preprocessing/vectorizer: lemmatization + TF-IDF")

    # Step 1: turn text into TF-IDF, then reduce it with SVD.
    X, vectorizer = make_vectors(products[text_column])
    cluster_features = reduce_vectors_for_clustering(X)

    # Step 2: scan k=2 to k=20 for both clustering methods.
    kmeans_df = scan_kmeans(cluster_features)
    hierarchical_df = scan_hierarchical(cluster_features)

    kmeans_df["scenario"] = scenario_name
    hierarchical_df["scenario"] = scenario_name

    kmeans_df.to_csv(METRICS / f"{scenario_name}_kmeans_k_scan.csv", index=False)
    hierarchical_df.to_csv(METRICS / f"{scenario_name}_hierarchical_k_scan.csv", index=False)

    plot_method_comparison(kmeans_df, hierarchical_df, scenario_name)

    # Step 3: keep the best KMeans setup for the main output.
    best_kmeans = kmeans_df.sort_values("silhouette_score", ascending=False).iloc[0]
    best_k = int(best_kmeans["k"])
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = final_model.fit_predict(cluster_features)

    # Step 4: save the main product list, cluster summary, and PCA plots.
    save_cluster_summary(products, X, vectorizer, final_labels, scenario_name)
    save_pca_plot(X, final_labels, scenario_name, products)

    if scenario_name == "after_canonical_no_reviews":
        # Step 5: extra presentation plots for easier-to-explain k=6 and k=7.
        for fixed_k in [6, 7]:
            kmeans_labels = fit_kmeans(cluster_features, k=fixed_k)
            save_cluster_summary(
                products,
                X,
                vectorizer,
                kmeans_labels,
                scenario_name,
                method_name="KMeans",
                output_tag=f"after_canonical_no_reviews_kmeans_{fixed_k}",
            )
            save_pca_plot(
                X,
                kmeans_labels,
                scenario_name,
                products,
                method_name="KMeans",
                output_tag=f"after_canonical_no_reviews_kmeans_{fixed_k}",
                color_mode="clusters",
            )

            hierarchical_labels = fit_hierarchical(cluster_features, k=fixed_k)
            save_cluster_summary(
                products,
                X,
                vectorizer,
                hierarchical_labels,
                scenario_name,
                method_name="Hierarchical",
                output_tag=f"after_canonical_no_reviews_hierarchical_{fixed_k}",
            )
            save_pca_plot(
                X,
                hierarchical_labels,
                scenario_name,
                products,
                method_name="Hierarchical",
                output_tag=f"after_canonical_no_reviews_hierarchical_{fixed_k}",
                color_mode="clusters",
            )

    best_hierarchical = hierarchical_df.sort_values("silhouette_score", ascending=False).iloc[0]
    comparison = [
        {
            "scenario": scenario_name,
            "method": "KMeans",
            "k": best_k,
            "inertia": best_kmeans["inertia"],
            "silhouette_score": best_kmeans["silhouette_score"],
        },
        {
            "scenario": scenario_name,
            "method": "Hierarchical",
            "k": int(best_hierarchical["k"]),
            "inertia": "",
            "silhouette_score": best_hierarchical["silhouette_score"],
        },
    ]

    print("Best KMeans:", f"k={best_k}", f"silhouette={best_kmeans['silhouette_score']:.3f}")
    return pd.DataFrame(comparison), kmeans_df, hierarchical_df


# ------------------------------------------------------------
# 8. Main script
# ------------------------------------------------------------


def main():
    print("Loading product table")
    products = load_product_table()
    print("Products:", len(products))

    # Scenario 1: noisy baseline with product + category + reviews.
    before_comparison, before_kmeans, before_hierarchical = run_scenario(
        products,
        text_column="before_text",
        scenario_name="before_with_reviews",
    )

    # Scenario 2: improved product identity only, no review text.
    after_comparison, after_kmeans, after_hierarchical = run_scenario(
        products,
        text_column="after_text",
        scenario_name="after_canonical_no_reviews",
    )

    # Save comparison tables for the presentation.
    all_comparison = pd.concat([before_comparison, after_comparison], ignore_index=True)
    all_comparison.to_csv(METRICS / "baseline_before_after_clustering_comparison.csv", index=False)

    pd.concat([before_kmeans, after_kmeans], ignore_index=True).to_csv(METRICS / "baseline_all_kmeans_k_scan.csv", index=False)
    pd.concat([before_hierarchical, after_hierarchical], ignore_index=True).to_csv(
        METRICS / "baseline_all_hierarchical_k_scan.csv",
        index=False,
    )

    print("\nBefore vs after clustering summary:")
    print(all_comparison[["scenario", "method", "k", "silhouette_score"]])

    apply_taxonomy_correction()
    save_product_groups_markdown()
    print("\nDone. Clustering outputs saved in outputs/metrics, outputs/figures, and outputs/clustering.")


if __name__ == "__main__":
    main()
