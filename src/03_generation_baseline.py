""" LLM-based generation using sentiment + clustering evidence.

Run:
    python src/03_generation_baseline.py

Quick test:
    python src/03_generation_baseline.py --sample-products 3

What this script does:
1. Loads the raw reviews.
2. Trains a simple lemmatization + TF-IDF + Linear SVM sentiment model.
3. Aggregates product-level evidence.
4. Joins the taxonomy-corrected clustering category.
5. Builds friendly prompts for Qwen/Ollama.
6. Saves Streamlit-ready product-card summaries.
"""

from pathlib import Path
import argparse
import ast
import json
import math
import re
import urllib.error
import urllib.request

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLUSTERING = ROOT / "outputs" / "clustering"
ARTICLES = ROOT / "outputs" / "articles"
PROMPTS = ARTICLES / "prompts"
MODELS = ROOT / "models"

for folder in [ARTICLES, PROMPTS, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)

EVIDENCE_CACHE = ARTICLES / "product_generation_evidence.csv"


# ============================================================
# 1. Load reviews and train the sentiment model
# ============================================================


def load_raw_reviews():
    """Load the raw review CSV files."""
    frames = []
    for path in RAW_DIR.glob("*.csv"):
        frame = pd.read_csv(path, low_memory=False)
        frame["source_file"] = path.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No CSV files found in data/raw.")

    return pd.concat(frames, ignore_index=True)


def make_review_table(raw):
    """Create the review table used by generation."""
    reviews = pd.DataFrame()
    reviews["product_name"] = raw.get("name", "").fillna("").astype(str)
    reviews["category_text"] = raw.get("categories", raw.get("primaryCategories", "")).fillna("").astype(str)
    reviews["review_title"] = raw.get("reviews.title", "").fillna("").astype(str)
    reviews["review_text"] = raw.get("reviews.text", "").fillna("").astype(str)
    reviews["rating"] = pd.to_numeric(raw.get("reviews.rating"), errors="coerce")
    reviews["text"] = (reviews["review_title"] + " " + reviews["review_text"]).str.strip()

    reviews = reviews.dropna(subset=["rating"]).copy()
    reviews = reviews[reviews["product_name"].str.len() > 0].copy()
    reviews = reviews[reviews["text"].str.len() > 0].copy()
    reviews = reviews.drop_duplicates(subset=["product_name", "review_title", "review_text", "rating"]).copy()

    reviews["rating_label"] = "positive"
    reviews.loc[reviews["rating"] <= 2, "rating_label"] = "negative"
    reviews.loc[reviews["rating"] == 3, "rating_label"] = "neutral"
    return reviews


def basic_clean(text):
    """Simple text cleaning."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
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


def train_sentiment_model(reviews):
    """Train the selected lemmatization + TF-IDF + Linear SVM model."""
    print("Training sentiment model: lemmatization + TF-IDF + Linear SVM")

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True)),
            ("svm", LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=10000)),
        ]
    )

    cleaned_text = reviews["text"].apply(lemmatize_text)
    model.fit(cleaned_text, reviews["rating_label"])
    reviews["predicted_sentiment"] = model.predict(cleaned_text)

    model_path = MODELS / "generation_lemmatization_tfidf_sentiment_model.joblib"
    try:
        import joblib

        joblib.dump(model, model_path)
        print(f"Saved generation sentiment model: {model_path}")
    except Exception:
        print("Could not save the generation sentiment model, but generation can continue.")

    return reviews


# ============================================================
# 2. Product-level evidence
#
# The LLM should not guess. We first create simple evidence:
# - review count
# - average rating
# - predicted sentiment shares
# - simple keyword-based strengths and concerns
# - corrected category from clustering
# ============================================================


STRENGTH_RULES = {
    "easy to use or set up": ["easy", "simple", "setup", "set up", "user friendly"],
    "clear screen or picture": ["screen", "display", "picture", "resolution", "clear"],
    "good for family use": ["kid", "kids", "child", "children", "family"],
    "useful for everyday needs": ["useful", "works", "handy", "daily", "everyday"],
    "portable or easy to carry": ["portable", "light", "lightweight", "carry"],
    "good basic build quality": ["quality", "solid", "sturdy", "build"],
    "familiar product format": ["battery", "batteries", "charger", "adapter", "cable", "case"],
}

COMPLAINT_RULES = {
    "setup or connection issues": ["wifi", "connect", "connection", "setup", "pairing"],
    "battery life concerns": ["battery", "drain", "charge", "charging"],
    "slow performance": ["slow", "lag", "freeze", "frozen"],
    "screen or display issues": ["screen", "display", "crack", "resolution"],
    "app or content limitations": ["app", "content", "ads", "advertisement"],
    "defective or dead-on-arrival units": ["defective", "dead", "broken", "stopped working"],
    "power or charging problems": ["power", "charger", "charging", "adapter"],
}


def find_themes(texts, rules, max_items=4):
    """Find simple repeated themes using keyword rules."""
    combined_text = " ".join(texts.fillna("").astype(str).str.lower())
    found = []

    for theme, keywords in rules.items():
        if any(keyword in combined_text for keyword in keywords):
            found.append(theme)

    return found[:max_items]


def load_taxonomy_groups():
    """Load the clustering taxonomy results."""
    taxonomy_path = CLUSTERING / "after_canonical_no_reviews_taxonomy_corrected_clusters.csv"
    if not taxonomy_path.exists():
        raise FileNotFoundError("Run python src/02_clustering_baseline.py first.")

    taxonomy = pd.read_csv(taxonomy_path)
    return taxonomy[
        ["product_name", "cluster_id", "corrected_category", "product_family", "correction_reason"]
    ].drop_duplicates("product_name")


def build_product_evidence(reviews):
    """Aggregate reviews into one row per product."""
    taxonomy = load_taxonomy_groups()
    rows = []

    for product_name, group in reviews.groupby("product_name"):
        predicted_counts = group["predicted_sentiment"].value_counts(normalize=True) * 100
        rating_counts = group["rating_label"].value_counts(normalize=True) * 100

        positive_reviews = group[group["predicted_sentiment"] == "positive"]["text"]
        negative_reviews = group[group["predicted_sentiment"] == "negative"]["text"]

        rows.append(
            {
                "product_name": product_name,
                "review_count": len(group),
                "average_rating": group["rating"].mean(),
                "positive_share": predicted_counts.get("positive", 0.0),
                "neutral_share": predicted_counts.get("neutral", 0.0),
                "negative_share": predicted_counts.get("negative", 0.0),
                "rating_positive_share": rating_counts.get("positive", 0.0),
                "strengths": find_themes(positive_reviews, STRENGTH_RULES),
                "concerns": find_themes(negative_reviews, COMPLAINT_RULES),
            }
        )

    products = pd.DataFrame(rows)
    products = products.merge(taxonomy, on="product_name", how="inner")
    products["evidence_strength"] = products["review_count"].apply(evidence_strength)

    # Same simple score idea as the original RoboReviews app:
    # 40% rating, 30% review volume, 20% positive sentiment, 10% risk control.
    log_reviews = products["review_count"].apply(lambda value: math.log(value + 1))
    if log_reviews.max() > log_reviews.min():
        products["review_volume_score"] = (log_reviews - log_reviews.min()) / (log_reviews.max() - log_reviews.min())
    else:
        products["review_volume_score"] = 1.0

    products["rating_score"] = (products["average_rating"] / 5).clip(0, 1)
    products["positive_sentiment_score"] = (products["positive_share"] / 100).clip(0, 1)
    products["risk_control_score"] = (1 - products["negative_share"] / 100).clip(0, 1)

    products["trust_score"] = (
        40 * products["rating_score"]
        + 30 * products["review_volume_score"]
        + 20 * products["positive_sentiment_score"]
        + 10 * products["risk_control_score"]
    ).round(1)

    products["rank_score"] = products["trust_score"]
    products = products.sort_values(["corrected_category", "rank_score"], ascending=[True, False])
    products["rank"] = products.groupby("corrected_category").cumcount() + 1
    return products


def evidence_strength(review_count):
    """Simple evidence labels used in the app."""
    if review_count >= 100:
        return "High"
    if review_count >= 30:
        return "Medium"
    return "Low"


def load_or_build_product_evidence(rebuild=False):
    """Use saved product evidence when available, like the original generation setup."""
    if EVIDENCE_CACHE.exists() and not rebuild:
        print(f"Loading saved product evidence: {EVIDENCE_CACHE}", flush=True)
        products = pd.read_csv(EVIDENCE_CACHE)
        for column in ["strengths", "concerns"]:
            if column in products.columns:
                products[column] = products[column].apply(parse_saved_list)
        return products

    print("Loading raw reviews", flush=True)
    reviews = make_review_table(load_raw_reviews())
    print("Review rows:", len(reviews), flush=True)

    reviews = train_sentiment_model(reviews)

    print("\nBuilding product evidence and joining taxonomy categories", flush=True)
    products = build_product_evidence(reviews)
    products.to_csv(EVIDENCE_CACHE, index=False)
    return products


def parse_saved_list(value):
    """Read list-like columns saved in CSV."""
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def clean_product_name(product_name):
    """Clean repeated product names for display, prompts, and saved cards."""
    text = str(product_name).replace("\r", "\n").strip()
    text = text.replace(",,,", "\n")
    text = text.replace("|", "\n")

    parts = []
    for part in text.splitlines():
        part = " ".join(part.split()).strip(" ,-\"")
        if part and part.lower() not in [old.lower() for old in parts]:
            parts.append(part)

    text = parts[0] if parts else str(product_name).strip()

    # Fix duplicated marketplace prefixes like "Amazon - Amazon Tap..."
    text = re.sub(r"(?i)^amazon\s*-\s*amazon\s+", "Amazon ", text)
    text = re.sub(r"(?i)^amazon\s+amazon\s+", "Amazon ", text)

    # Make common Amazon product names easier to read.
    text = re.sub(r"(?i)^amazon\s*-\s*(kindle|fire|echo)", r"Amazon \1", text)
    text = re.sub(r"\s+", " ", text).strip(" ,-\"")
    text = text.replace(" Fire Hd ", " Fire HD ")
    text = text.replace(" Hd ", " HD ")
    return text


# ============================================================
# 3. Prompt building
#
# The prompt is the most important part of generation quality.
# It gives Qwen:
# - compact evidence
# - product type
# - a short example
# - strict output labels that are easy to parse later
# ============================================================


def product_type_from_category(category, product_name):
    """Give the LLM a simple product type."""
    text = f"{category} {product_name}".lower()

    if "fire tv" in text or "streaming" in text:
        return "streaming product"
    if "echo" in text or "alexa" in text or "speaker" in text:
        return "smart speaker product"
    if "kindle" in text or "e-reader" in text or "ereader" in text:
        return "e-reader"
    if "fire tablet" in text or "tablet" in text:
        return "tablet"
    if "battery" in text or "charger" in text or "adapter" in text or "cable" in text or "case" in text:
        return "electronic accessory"
    if "non" in category.lower():
        return "non-electronic product"
    return "consumer product"


def comparison_note(product, category_products):
    """Create a simple category comparison note."""
    median_reviews = category_products["review_count"].median()
    median_rating = category_products["average_rating"].median()

    notes = []
    if product["review_count"] >= median_reviews:
        notes.append("review volume is above or around the category middle")
    else:
        notes.append("review volume is below the category middle")

    if product["average_rating"] >= median_rating:
        notes.append("average rating is strong compared with similar listed products")
    else:
        notes.append("average rating is still useful but below some similar listed products")

    if product["negative_share"] >= category_products["negative_share"].median():
        notes.append("complaint signals are worth comparing carefully")
    else:
        notes.append("complaint signals are lower than several similar listed products")

    return notes


def filter_strengths_for_product(product_name, product_type, strengths):
    """Remove strengths that do not fit the product identity."""
    product_text = str(product_name).lower()
    filtered = []

    if "battery" in product_text or "batteries" in product_text:
        allowed = [
            "useful for everyday needs",
            "familiar product format",
            "good basic build quality",
        ]
        filtered = [strength for strength in strengths if strength in allowed]
        return (filtered or ["useful for everyday battery needs"])[:4]

    if product_type == "electronic accessory":
        blocked_words = ["screen", "picture", "display", "family"]
        filtered = [
            strength for strength in strengths
            if not any(word in str(strength).lower() for word in blocked_words)
        ]
        return (filtered or ["useful for this accessory type"])[:4]

    for strength in strengths:
        strength_text = str(strength).lower()
        screen_strength = "screen" in strength_text or "picture" in strength_text or "display" in strength_text
        has_screen_in_name = any(word in product_text for word in ["show", "screen", "display", "tablet", "kindle", "e-reader"])

        if screen_strength and product_type == "smart speaker product" and not has_screen_in_name:
            continue

        filtered.append(strength)

    if not filtered:
        filtered = ["useful for this product type"]

    return filtered[:4]


def filter_concerns_for_product(product_name, product_type, concerns):
    """Remove complaint themes that do not fit the product identity."""
    product_text = str(product_name).lower()

    if "battery" in product_text or "batteries" in product_text:
        allowed = [
            "battery life concerns",
            "defective or dead-on-arrival units",
        ]
        filtered = [concern for concern in concerns if concern in allowed]
        return (filtered or ["some customers question battery life or consistency"])[:4]

    if product_type == "electronic accessory":
        blocked_words = ["screen", "display", "app", "content", "slow performance"]
        filtered = [
            concern for concern in concerns
            if not any(word in str(concern).lower() for word in blocked_words)
        ]
        return (filtered or ["reliability concerns may matter for some buyers"])[:4]

    return concerns[:4]


def get_style_angle(product_name, rank):
    """Change the writing rhythm from product to product."""
    angles = [
        "Lead with the kind of shopper this product fits.",
        "Lead with why the product feels easy to live with.",
        "Lead with the product's practical role in a normal buying decision.",
        "Lead with the balance between appeal and possible concerns.",
        "Lead with what makes the product a sensible shortlist candidate.",
    ]
    index = sum(ord(char) for char in f"{product_name}-{rank}") % len(angles)
    return angles[index]


def few_shot_example(product_type):
    """Few-shot example copied from the final prompt style we liked."""
    if product_type == "smart speaker product":
        return """Example:
HEADLINE: A Friendly Everyday Speaker Pick

SUMMARY:
Echo (White) feels like the kind of speaker people choose when they want something familiar, simple, and easy to live with. It has 3,309 reviews, averages 4.65/5, and 94.3% of reviews are positive. That gives the recommendation a stronger base than products with only a handful of opinions.

It should suit shoppers who want a regular Echo or Alexa-style speaker for everyday household use. Customers mainly respond well to the easy setup, the straightforward daily usefulness, and the way it works comfortably in typical home settings. It is not trying to be fancy; the appeal is that it sounds like a simple product people can start using without much friction.

The main concerns are connection issues, app or content limitations, and charging or power performance. Compared with similar Echo or Alexa speakers, it is widely reviewed and strongly rated, but those repeated concerns still matter. Overall, it is worth considering if you want a familiar Echo-style speaker and are comfortable with those possible issues.

BEST_FOR: Choose this if you want a widely reviewed Echo-style speaker for regular household use.

PROS:
- Easy setup
- Useful for everyday household use
- Works well in typical home settings

CONS:
- Connection issues
- App or content limitations
- Charging or power concerns"""

    if product_type == "tablet":
        return """Example for a Fire tablet:
HEADLINE: A Straightforward Tablet For Everyday Use

SUMMARY:
This Fire tablet is for shoppers who want a familiar screen for basic everyday use without making the choice feel complicated. It has 2,814 reviews, averages 4.59/5, and 94.7% of reviews are positive. That gives the product a strong base, although the repeated concerns should still be part of the decision.

It should suit people who want an easy-to-use tablet for regular home or family use. Customers mainly respond well to the simple setup, clear screen or picture, and the way it fits everyday use. The appeal is not that it sounds like a premium device; it is that many customers describe the basic experience positively.

The main concerns are app or content limitations, charging or power issues, battery life concerns, and screen or display problems. Compared with similar Fire tablets, it has a strong review base and a strong rating, but it is not free from friction. Overall, it is worth considering if you want a simple tablet and can accept those possible trade-offs.

BEST_FOR: Choose this if you want a straightforward tablet for regular everyday use.

PROS:
- Easy to use or set up
- Clear screen or picture
- Good for family use

CONS:
- App or content limitations
- Charging or power issues
- Battery life concerns"""

    if product_type == "e-reader":
        return """Example for a Kindle or e-reader:
HEADLINE: A Calm Reading Pick With Strong Appeal

SUMMARY:
This Kindle-style e-reader is for shoppers who want a focused reading device that feels simple, portable, and easy to return to. It has 1,240 reviews, averages 4.58/5, and 92.0% of reviews are positive. That gives the recommendation a strong base, while still leaving room to consider repeated concerns.

It should suit readers who want a device built around straightforward reading rather than a busy all-purpose screen. Customers mainly respond well to the clear display, easy setup, lightweight feel, and everyday usefulness. The appeal is that it sounds like a comfortable reading companion: simple enough for regular use, but still supported by enough reviews to make the signal useful.

The main concerns are battery life, setup or connection problems, screen or display issues, and app or content limitations. Compared with similar e-readers, it is strongly rated, but not every complaint disappears. Overall, it is worth considering if you want a reading-focused device and can accept those possible issues.

BEST_FOR: Choose this if you want a simple, reading-focused device.

PROS:
- Clear screen or picture
- Lightweight and portable
- Easy to use or set up

CONS:
- Battery life concerns
- Setup or connection problems
- Screen or display issues"""

    return """Example for an accessory or simple product:
HEADLINE: A Practical Everyday Backup Pick

SUMMARY:
This accessory product is for shoppers who want something simple to keep on hand rather than something they need to think about every day. It has 3,519 reviews, averages 4.43/5, and 86.0% of reviews are positive. That gives the recommendation a useful base, especially compared with products with only a few opinions.

It should suit people who want a straightforward item for regular household or device-related use. Customers mainly respond well to the familiar format, everyday usefulness, and sense that the product does its basic job without much fuss. The appeal is practical rather than flashy: it is the type of product people buy because they need the category.

The main concerns are battery life, defective units, charging or power issues, and return or replacement problems. Compared with similar accessories, it can have a strong review base, but repeated concerns still matter. Overall, it is worth considering if you need this product type and are comfortable comparing alternatives when reliability concerns are important.

BEST_FOR: Choose this if you need a practical everyday accessory with many reviews behind it.

PROS:
- Familiar product format
- Useful for everyday needs
- Strong review volume

CONS:
- Battery life concerns
- Defective units
- Power-related issues"""


def build_prompt(product, category_products):
    """Build the final few-shot prompt for one product card."""
    display_name = clean_product_name(product["product_name"])
    product_type = product_type_from_category(product["corrected_category"], display_name)
    strengths = product["strengths"] if product["strengths"] else ["useful for this product type"]
    strengths = filter_strengths_for_product(display_name, product_type, strengths)
    concerns = product["concerns"] if product["concerns"] else ["limited repeated complaint themes"]
    concerns = filter_concerns_for_product(display_name, product_type, concerns)
    notes = comparison_note(product, category_products)

    compact_evidence = {
        "product_name": display_name,
        "category": product["corrected_category"],
        "product_type": product_type,
        "review_count": f"{int(product['review_count']):,}",
        "average_rating": f"{float(product['average_rating']):.2f}/5",
        "positive_review_share": f"{float(product['positive_share']):.1f}%",
        "negative_review_share": f"{float(product['negative_share']):.1f}%",
        "evidence_strength": product["evidence_strength"],
        "main_strengths": strengths[:4],
        "main_concerns": concerns[:4],
        "comparison_notes": notes,
        "recommendation": "worth considering if the listed concerns are acceptable",
        "style_angle": get_style_angle(display_name, product["rank"]),
    }

    required_sentence = (
        f"It has {compact_evidence['review_count']} reviews, "
        f"averages {compact_evidence['average_rating']}, and "
        f"{compact_evidence['positive_review_share']} of reviews are positive."
    )

    return f"""You are a friendly consumer product reviewer writing for everyday shoppers.

Your task:
Rewrite the product evidence into a natural review.

Return only this labeled format. Do not use JSON. Do not skip any label.

HEADLINE: 1 friendly reviewer-style headline, maximum 12 words

SUMMARY:
The main mini-review. Write exactly 3 paragraphs. Each paragraph should be 60-75 words, for 190-220 words total. Use a blank line between paragraphs. Do not write only one or two paragraphs.

BEST_FOR: 1 short customer-fit sentence, maximum 22 words

PROS:
- max 3 short evidence-based pros

CONS:
- max 3 short evidence-based cons or watch-outs

Rules:
- Use simple, smooth language.
- Be friendly, lightly creative, and human.
- Do not sound like a data report.
- Do not start with "This is a..." or "[Product] is a ... product."
- You may be creative in wording, rhythm, and reviewer-style phrasing. Do not be creative with facts.
- Use style_angle to vary the narration from product to product while keeping the same evidence backbone.
- Do not say "positive customer feedback", "positive review text", "negative text rate", "watch-outs", "complaint risk", "review support", "sample", "metric", "dataset", or "comparison group".
- Do not invent facts.
- Use the exact review_count, average_rating, and positive_review_share from compact evidence.
- Mention the review numbers only once, in paragraph 1, using the required evidence sentence exactly.
- Mention both strengths and complaints.
- Use only product_type, main_strengths, main_concerns, comparison_notes, evidence_strength, and recommendation.
- Do not mention outside competitors, outside brands, or products not present in the evidence.
- If the rating is above 4.3/5, do not make it sound low.
- Do not mention price, affordability, warranty, guaranteed durability, return rates, "best", "perfect", "no issues", or "no complaints".
- For Low evidence, clearly say the evidence is limited and keep the recommendation cautious.
- End paragraph 3 with a clear buying recommendation.
- Keep paragraph spacing consistent: exactly one blank line between the 3 summary paragraphs.
- PROS must have no more than 3 bullets.
- CONS must have no more than 3 bullets.

Few-shot example for style and structure. Do not copy the exact product name, numbers, or complaint phrases unless they match the real evidence.

{few_shot_example(product_type)}

Required evidence sentence:
{required_sentence}

Now write the product card for the compact evidence below.

Compact evidence:
{json.dumps(compact_evidence, indent=2)}
"""


# ============================================================
# 4. Call Ollama and parse the answer
# ============================================================


def call_ollama(prompt, model_name):
    """Send one prompt to local Ollama/Qwen and return the text answer."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You write clear, friendly, evidence-grounded product reviews."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.65,
            "top_p": 0.90,
            "repeat_penalty": 1.12,
            "num_predict": 580,
        },
    }
    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result.get("message", {}).get("content", "").strip()
    except urllib.error.URLError as error:
        return f"Ollama generation failed. Start Ollama and run `ollama pull {model_name}`. Error: {error}"
    except Exception as error:
        return f"Ollama generation failed. Error: {error}"


def parse_card(text):
    """Turn the labeled LLM answer into a normal Python dictionary."""
    card = {"headline": "", "summary": "", "best_for": "", "pros": [], "cons": []}
    current_label = None

    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("HEADLINE:"):
            card["headline"] = stripped.replace("HEADLINE:", "").strip()
            current_label = "headline"
        elif stripped.startswith("SUMMARY:"):
            current_label = "summary"
        elif stripped.startswith("BEST_FOR:"):
            card["best_for"] = stripped.replace("BEST_FOR:", "").strip()
            current_label = "best_for"
        elif stripped.startswith("PROS:"):
            current_label = "pros"
        elif stripped.startswith("CONS:"):
            current_label = "cons"
        elif current_label == "summary":
            card["summary"] += line + "\n"
        elif current_label in ["pros", "cons"] and stripped.startswith("-"):
            card[current_label].append(stripped[1:].strip())

    card["summary"] = card["summary"].strip()
    card["pros"] = card["pros"][:3]
    card["cons"] = card["cons"][:3]
    return card


def simple_non_llm_card(product, category_products):
    """A backup card if Ollama is not running."""
    display_name = clean_product_name(product["product_name"])
    strengths = product["strengths"] if product["strengths"] else ["useful for this product type"]
    concerns = product["concerns"] if product["concerns"] else ["limited repeated complaint themes"]
    notes = comparison_note(product, category_products)
    product_type = product_type_from_category(product["corrected_category"], display_name)
    strengths = filter_strengths_for_product(display_name, product_type, strengths)
    concerns = filter_concerns_for_product(display_name, product_type, concerns)

    summary = (
        f"{display_name} is a {product_type} for shoppers comparing options in "
        f"{product['corrected_category']}. It has {int(product['review_count']):,} reviews, "
        f"averages {product['average_rating']:.2f}/5, and {product['positive_share']:.1f}% of reviews are positive.\n\n"
        f"Customers mainly respond well to {', '.join(strengths[:3])}. "
        f"This gives the product a useful starting point, especially when the buyer wants something practical and easy to compare.\n\n"
        f"The main concerns are {', '.join(concerns[:3])}. Compared with similar listed products, "
        f"{'; '.join(notes)}. Overall, it is worth considering if those concerns are acceptable."
    )

    return {
        "headline": "Evidence-Based Shopper Review",
        "summary": summary,
        "best_for": "Choose this if the product type fits your need and the concerns are acceptable.",
        "pros": strengths[:3],
        "cons": concerns[:3],
        "source": "simple_backup_no_llm",
    }


# ============================================================
# 5. Generate and save product cards
# ============================================================


def slugify(text):
    text = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return text[:80] or "product"


def make_card_key(category, name):
    """Key used to avoid regenerating existing cards."""
    return (str(category).strip().lower(), clean_product_name(name).lower())


def load_existing_cards(card_path):
    """Load previous card output so a long run can resume."""
    if not card_path.exists():
        return [], {}

    cards = json.loads(card_path.read_text(encoding="utf-8"))
    lookup = {
        make_card_key(card.get("category", ""), card.get("name", "")): card
        for card in cards
    }
    return cards, lookup


def generate_cards(products, llm_model, sample_products=None, skip_llm=False, only_missing=False):
    """Generate product-card summaries."""

    # Step 1: keep one clean card per product/category.
    products_to_generate = products.copy()
    products_to_generate["display_name"] = products_to_generate["product_name"].apply(clean_product_name)
    products_to_generate = products_to_generate.sort_values(
        ["corrected_category", "display_name", "rank_score", "review_count"],
        ascending=[True, True, False, False],
    )
    products_to_generate = products_to_generate.drop_duplicates(
        subset=["corrected_category", "display_name"],
        keep="first",
    ).copy()
    products_to_generate = products_to_generate.sort_values(["corrected_category", "rank_score"], ascending=[True, False])
    products_to_generate["rank"] = products_to_generate.groupby("corrected_category").cumcount() + 1

    if sample_products:
        products_to_generate = products_to_generate.head(sample_products)

    # Step 2: optionally continue from the existing JSON file.
    card_path = ARTICLES / "product_card_summaries.json"
    cards, existing_lookup = load_existing_cards(card_path) if only_missing else ([], {})

    # Step 3: decide which products really need generation.
    selected_rows = []
    for _, product in products_to_generate.iterrows():
        display_name = clean_product_name(product["product_name"])
        key = make_card_key(product["corrected_category"], display_name)
        if only_missing and key in existing_lookup:
            continue
        selected_rows.append(product)

    total = len(selected_rows)
    if total == 0:
        print("No missing product cards to generate.", flush=True)
        return cards

    # Step 4: generate one card at a time and save progress after every product.
    for index, product in enumerate(selected_rows, start=1):
        display_name = clean_product_name(product["product_name"])
        category_products = products[products["corrected_category"] == product["corrected_category"]]
        prompt = build_prompt(product, category_products)
        prompt_path = PROMPTS / f"{slugify(product['corrected_category'])}_{slugify(display_name)}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        filled = int(index / total * 24)
        bar = "#" * filled + "-" * (24 - filled)
        print(f"\n[{bar}] {index}/{total}: {display_name[:90]}", flush=True)

        if skip_llm:
            parsed = simple_non_llm_card(product, category_products)
            raw_output = ""
        else:
            raw_output = call_ollama(prompt, llm_model)
            parsed = parse_card(raw_output)
            parsed["source"] = "ollama_qwen"
            if not parsed["summary"] or raw_output.startswith("Ollama generation failed"):
                parsed = simple_non_llm_card(product, category_products)
                parsed["llm_error"] = raw_output

        parsed.update(
            {
                "category": product["corrected_category"],
                "rank": int(product["rank"]),
                "name": display_name,
                "raw_product_name": product["product_name"],
                "review_count": int(product["review_count"]),
                "average_rating": round(float(product["average_rating"]), 3),
                "positive_share": round(float(product["positive_share"]), 1),
                "negative_share": round(float(product["negative_share"]), 1),
                "trust_score": round(float(product["trust_score"]), 1),
                "rating_score": round(float(product["rating_score"]), 3),
                "review_volume_score": round(float(product["review_volume_score"]), 3),
                "positive_sentiment_score": round(float(product["positive_sentiment_score"]), 3),
                "risk_control_score": round(float(product["risk_control_score"]), 3),
                "evidence_strength": product["evidence_strength"],
                "cluster_id": int(product["cluster_id"]),
                "product_family": product.get("product_family", ""),
                "prompt_file": str(prompt_path.relative_to(ROOT)),
            }
        )
        cards.append(parsed)
        card_path.write_text(json.dumps(cards, indent=2), encoding="utf-8")
        print(f"Saved progress to {card_path}", flush=True)

    return cards


def save_markdown(cards):
    """Save a readable markdown version of the product cards."""
    lines = ["# Generated Product Card Reviews\n"]

    for card in cards:
        lines.append(f"## {card['category']} | #{card['rank']} | {card['name']}\n")
        lines.append(f"**{card['headline']}**\n")
        lines.append(card["summary"])
        lines.append("")
        lines.append(f"**Best for:** {card['best_for']}")
        lines.append("")
        lines.append("**Pros:**")
        for pro in card["pros"]:
            lines.append(f"- {pro}")
        lines.append("")
        lines.append("**Cons:**")
        for con in card["cons"]:
            lines.append(f"- {con}")
        lines.append("\n---\n")

    output_path = ARTICLES / "product_card_summaries.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_streamlit_json(cards, products):
    """Save JSON files that a Streamlit app can read."""
    card_path = ARTICLES / "product_card_summaries.json"
    evidence_path = ARTICLES / "product_evidence_for_streamlit.csv"

    card_path.write_text(json.dumps(cards, indent=2), encoding="utf-8")
    products.to_csv(evidence_path, index=False)
    return card_path, evidence_path


# ------------------------------------------------------------
# 6. Main
# ------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-model", default="qwen2.5:3b")
    parser.add_argument("--sample-products", type=int, help="Generate only the first N products for a quick test.")
    parser.add_argument("--skip-llm", action="store_true", help="Build evidence and backup cards without calling Ollama.")
    parser.add_argument("--only-missing", action="store_true", help="Keep existing cards and generate only missing products.")
    parser.add_argument("--rebuild-evidence", action="store_true", help="Rebuild product evidence from raw reviews.")
    args = parser.parse_args()

    products = load_or_build_product_evidence(rebuild=args.rebuild_evidence)
    print("Products with taxonomy category:", len(products))
    print(products["corrected_category"].value_counts())

    print("\nGenerating Streamlit-ready product cards")
    cards = generate_cards(
        products,
        llm_model=args.llm_model,
        sample_products=args.sample_products,
        skip_llm=args.skip_llm,
        only_missing=args.only_missing,
    )

    card_path, evidence_path = save_streamlit_json(cards, products)
    markdown_path = save_markdown(cards)

    print("\nDone. Generation outputs saved:")
    print(card_path)
    print(evidence_path)
    print(markdown_path)
    print(PROMPTS)


if __name__ == "__main__":
    main()
