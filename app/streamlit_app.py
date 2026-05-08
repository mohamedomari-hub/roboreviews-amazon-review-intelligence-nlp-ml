"""Streamlit app for the clean baseline copy.

Run:
    streamlit run app/streamlit_app.py

Before running the app, generate product cards with:
    python src/03_generation_baseline.py --sample-products 10
or:
    python src/03_generation_baseline.py
"""

from pathlib import Path
import html
import json
import re
import textwrap

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OLD_PROJECT_RAW_DIR = ROOT.parent / "amazon-review-intelligence" / "data" / "raw"
CARDS_PATH = ROOT / "outputs" / "articles" / "product_card_summaries.json"
EVIDENCE_PATH = ROOT / "outputs" / "articles" / "product_evidence_for_streamlit.csv"

IMAGE_ALIAS_NAMES = {
    "echo white": [
        "Amazon Echo ‚Äì White",
        "Amazon Echo (1st Generationcertified) Color:White Free Shipping",
    ],
    "echo black": [
        "Certified Refurbished Amazon Echo",
        "Echo Dot (Previous generation)",
    ],
    "amazon amazon tap portable bluetooth and wi fi speaker": [
        "Amazon Tap - Alexa-Enabled Portable Bluetooth Speaker",
        "Amazon Tap Smart Assistant Alexaenabled (black) Brand New",
    ],
    "amazon tap portable bluetooth and wi fi speaker black": [
        "Amazon Tap - Alexa-Enabled Portable Bluetooth Speaker",
        "Amazon Tap Smart Assistant Alexaenabled (black) Brand New",
    ],
    "amazon fire tv": [
        "Amazon Fire TV with 4K Ultra HD and Alexa Voice Remote (Pendant Design) | Streaming Media Player",
        "Amazon Fire TV Gaming Edition Streaming Media Player",
        "Certified Refurbished Amazon Fire TV with Alexa Voice Remote",
    ],
    "amazon echo and fire tv power adapter": [
        "Amazon 9W PowerFast Official OEM USB Charger and Power Adapter for Fire Tablets and Kindle eReaders",
        "Amazon Kindle Charger Power Adapter Wall Charger And Usb Cable Micro Usb Cord",
    ],
    "certified refurbished amazon fire tv stick": [
        "Fire TV Stick Streaming Media Player Pair Kit",
    ],
    "amazon fire hd 8 8in tablet 16gb b018szt3bk 6th gen android": [
        "All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 16 GB - Includes Special Offers, Black",
        "Amazon Fire HD 8 with Alexa (8\" HD Display Tablet)",
    ],
    "kindle paperwhite": [
        "All-New Kindle E-reader - Black, 6\" Glare-Free Touchscreen Display, Wi-Fi - Includes Special Offers",
        "Amazon Kindle E-Reader 6\" Wifi (8th Generation, 2016)",
    ],
    "amazon kindle paperwhite ebook reader 4 gb 6 monochrome paperwhite touchscreen wi fi black": [
        "All-New Kindle E-reader - Black, 6\" Glare-Free Touchscreen Display, Wi-Fi - Includes Special Offers",
        "Amazon Kindle E-Reader 6\" Wifi (8th Generation, 2016)",
    ],
    "kindle keyboard": [
        "AmazonBasics Bluetooth Keyboard for Android Devices - Black",
    ],
    "amazon kindle fire hd 3rd gen 8gb": [
        "All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 16 GB - Includes Special Offers, Black",
        "Fire Tablet, 7 Display, Wi-Fi, 8 GB - Includes Special Offers, Black",
    ],
    "certified refurbished amazon fire tv 1st": [
        "Certified Refurbished Amazon Fire TV with Alexa Voice Remote",
        "Amazon Fire TV Gaming Edition Streaming Media Player",
    ],
}


st.set_page_config(page_title="RoboReviews", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fbfaf7 0%, #f4f8f6 42%, #ffffff 100%);
        color: #17212b;
    }
    header[data-testid="stHeader"] {
        background: transparent;
        height: 0;
    }
    header[data-testid="stHeader"] * {
        display: none;
    }
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    .stDeployButton,
    #MainMenu {
        display: none;
        visibility: hidden;
    }
    .block-container {
        padding-top: 0;
        padding-bottom: 3rem;
        max-width: 1480px;
    }
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(150deg, rgba(224, 100, 85, 0.18) 0%, rgba(224, 100, 85, 0.03) 30%, transparent 31%),
            linear-gradient(180deg, #17212b 0%, #0f5f59 58%, #d97706 135%);
        border-right: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 10px 0 34px rgba(15, 33, 43, 0.16);
        margin-top: 0;
        border-top-right-radius: 22px;
        overflow: hidden;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #fff7ed;
        letter-spacing: 0;
    }
    section[data-testid="stSidebar"] label {
        color: #ecfdf5;
        font-weight: 800;
        letter-spacing: 0;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] input {
        border-color: rgba(255, 255, 255, 0.32);
        background: rgba(255, 253, 250, 0.96);
        border-radius: 13px;
        box-shadow: 0 10px 24px rgba(15, 33, 43, 0.14);
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within,
    section[data-testid="stSidebar"] input:focus {
        border-color: #fed7aa;
        box-shadow: 0 0 0 3px rgba(254, 215, 170, 0.28);
    }
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {
        color: #fff7ed;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #fed7aa;
        border-color: #fff7ed;
        box-shadow: 0 0 0 4px rgba(254, 215, 170, 0.24);
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #d7f7ef;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #e7ddd3;
        border-radius: 18px;
        box-shadow: 0 16px 38px rgba(43, 34, 28, 0.08);
        padding: 1.2rem 1.2rem 1rem 1.2rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #d7c7ba;
        box-shadow: 0 18px 44px rgba(43, 34, 28, 0.11);
        transform: translateY(-1px);
        transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
    }
    .rr-hero {
        border-radius: 22px;
        padding: 1.35rem 1.65rem;
        margin-bottom: 1.5rem;
        margin-top: -1.35rem;
        margin-left: 0;
        width: 100%;
        background:
            linear-gradient(150deg, rgba(224, 100, 85, 0.18) 0%, rgba(224, 100, 85, 0.03) 30%, transparent 31%),
            linear-gradient(90deg, #17212b 0%, #0f5f59 46%, #0f766e 70%, #d97706 118%);
        color: white;
        border-left: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: 0 18px 45px rgba(15, 33, 43, 0.22);
        position: relative;
        overflow: hidden;
    }
    .rr-hero::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 10px;
        background: linear-gradient(180deg, rgba(254, 215, 170, 0.85), rgba(224, 100, 85, 0.18));
    }
    .rr-hero > * {
        position: relative;
        z-index: 1;
    }
    .rr-hero-kicker {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        color: #fed7aa;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .rr-hero-title {
        font-size: clamp(2rem, 3.2vw, 3.15rem);
        font-weight: 850;
        letter-spacing: 0;
        line-height: 1.02;
        margin-bottom: 0.45rem;
    }
    .rr-hero-subtitle {
        font-size: 1rem;
        line-height: 1.45;
        max-width: 760px;
        color: #effaf5;
    }
    .rr-section-title {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 0.35rem 0 0.3rem 0;
    }
    .rr-section-title h2 {
        font-size: 1.45rem;
        margin: 0;
        color: #17212b;
        letter-spacing: 0;
    }
    .rr-category-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 750;
        color: #0f5f59;
        background: #e7f7f2;
        border: 1px solid #a7d8ce;
    }
    .rr-card-title {
        font-size: 1.08rem;
        font-weight: 820;
        line-height: 1.25;
        color: #17212b;
        margin-top: 0.55rem;
        min-height: 4.6rem;
    }
    .rr-pick-label {
        display: inline-flex;
        width: fit-content;
        border-radius: 999px;
        padding: 0.34rem 0.7rem;
        background: #fff1e7;
        border: 1px solid #fed7aa;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #9a3412;
        margin-bottom: 0.7rem;
    }
    .rr-image-placeholder {
        height: 220px;
        border-radius: 14px;
        border: 1px dashed #e9d8c9;
        background: #ffffff;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 1rem;
        color: #5f6b76;
        margin-bottom: 0.6rem;
    }
    .rr-image-placeholder-title {
        font-size: 0.86rem;
        font-weight: 820;
        color: #17212b;
        margin-bottom: 0.25rem;
    }
    .rr-image-placeholder-copy {
        font-size: 0.76rem;
        line-height: 1.35;
        max-width: 220px;
    }
    div[data-testid="stImage"] {
        height: 220px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border-radius: 14px;
        background: #ffffff;
        border: 1px solid #efe5dc;
        margin-bottom: 0.55rem;
    }
    div[data-testid="stImage"] img {
        width: 100%;
        height: 220px;
        object-fit: contain;
    }
    .rr-verdict {
        font-size: 0.98rem;
        line-height: 1.35;
        min-height: 3.2rem;
        color: #0f5f59;
        font-weight: 760;
        margin-bottom: 0.35rem;
    }
    .rr-blog {
        font-size: 0.93rem;
        line-height: 1.55;
        color: #364553;
        margin: 0.45rem 0 1rem 0;
        height: 12.5rem;
        overflow-y: scroll;
        padding: 0.1rem 0.7rem 0.1rem 0;
        scrollbar-width: thin;
        scrollbar-color: #0f766e #e7f7f2;
    }
    .rr-blog::-webkit-scrollbar {
        width: 10px;
    }
    .rr-blog::-webkit-scrollbar-track {
        background: #e7f7f2;
        border-radius: 999px;
        border: 1px solid #c8e7df;
    }
    .rr-blog::-webkit-scrollbar-thumb {
        background: #0f766e;
        border-radius: 999px;
        border: 2px solid #e7f7f2;
    }
    .rr-blog::-webkit-scrollbar-thumb:hover {
        background: #0f5f59;
    }
    .rr-blog p {
        margin: 0 0 0.78rem 0;
    }
    .rr-blog p:last-child {
        margin-bottom: 0;
    }
    .rr-stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.55rem;
        margin: 0.75rem 0 0.95rem 0;
    }
    .rr-stat {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.58rem 0.64rem;
        background: linear-gradient(180deg, #ffffff 0%, #fbf9f6 100%);
        min-width: 0;
    }
    .rr-stat-label {
        font-size: 0.68rem;
        color: #7a6a5e;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 800;
        margin-bottom: 0.18rem;
        white-space: nowrap;
    }
    .rr-stat-value {
        font-size: clamp(1rem, 1.4vw, 1.22rem);
        color: #17212b;
        font-weight: 850;
        letter-spacing: 0;
        white-space: nowrap;
        overflow: visible;
    }
    .rr-evidence-high .rr-stat-value { color: #0f766e; }
    .rr-evidence-medium .rr-stat-value { color: #c56a13; }
    .rr-evidence-low .rr-stat-value { color: #b43b32; }
    .rr-card-subhead {
        color: #17212b;
        font-size: 0.86rem;
        font-weight: 820;
        margin: 0.8rem 0 0.25rem 0;
    }
    .rr-fit-box {
        border-left: 4px solid #0f766e;
        background: #eef9f5;
        color: #154d48;
        padding: 0.65rem 0.72rem;
        border-radius: 10px;
        font-size: 0.86rem;
        line-height: 1.4;
        margin-bottom: 0.75rem;
    }
    .rr-list-item {
        border-radius: 10px;
        padding: 0.46rem 0.58rem;
        margin: 0.28rem 0;
        font-size: 0.84rem;
        line-height: 1.34;
    }
    .rr-pro {
        color: #116149;
        background: #edf9f1;
        border: 1px solid #b7e4c7;
    }
    .rr-con {
        color: #9b2c22;
        background: #fff0ed;
        border: 1px solid #f6c5bd;
    }
    .rr-source {
        display: inline-flex;
        margin-top: 0.65rem;
        border-radius: 999px;
        padding: 0.24rem 0.55rem;
        background: #f5eee6;
        color: #7a6a5e;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .rr-muted {
        color: #726a63;
    }
    .rr-sidebar-note {
        margin-top: 1rem;
        border-radius: 16px;
        padding: 0.95rem 1rem;
        background: rgba(255, 253, 250, 0.94);
        border: 1px solid rgba(254, 215, 170, 0.68);
        color: #263442;
        font-size: 0.84rem;
        line-height: 1.45;
        box-shadow: 0 18px 38px rgba(15, 33, 43, 0.20);
    }
    .rr-sidebar-note strong {
        color: #9a3412;
        font-weight: 850;
    }
    .rr-sidebar-logo {
        margin: -3.6rem 0 1.45rem 0;
        padding: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .rr-sidebar-logo-mark {
        width: 155px;
        height: 112px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0;
    }
    .rr-sidebar-logo-mark svg {
        width: 100%;
        height: 100%;
        filter: drop-shadow(0 16px 22px rgba(0, 0, 0, 0.24));
    }
    .rr-details {
        margin-top: 0.65rem;
        border: 1px solid #d9ece6;
        border-radius: 14px;
        background: #fbfefd;
        overflow: hidden;
    }
    .rr-details summary {
        cursor: pointer;
        list-style: none;
        padding: 0.72rem 0.85rem;
        color: #0f5f59;
        font-size: 0.88rem;
        font-weight: 850;
        background: linear-gradient(180deg, #eef9f5 0%, #fbfefd 100%);
    }
    .rr-details summary::-webkit-details-marker {
        display: none;
    }
    .rr-details summary::after {
        content: "⌄";
        float: right;
        color: #0f766e;
        font-size: 1rem;
        line-height: 1;
    }
    .rr-details[open] summary::after {
        content: "⌃";
    }
    .rr-details-body {
        padding: 0.05rem 0.75rem 0.75rem 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def escape_text(value):
    return html.escape(str(value), quote=True)


def render_paragraphs(value):
    text = str(value or "").strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return "".join(f"<p>{escape_text(paragraph)}</p>" for paragraph in paragraphs)


def render_html(markup):
    """Render custom HTML without Markdown turning indentation into code blocks."""
    st.markdown(textwrap.dedent(str(markup)).strip(), unsafe_allow_html=True)


def safe_list(value):
    return value if isinstance(value, list) else []


def build_fallback_review(card):
    """Create a simple review if the generated summary is missing."""
    name = clean_product_name(card.get("name", "This product"))
    rating = float(card.get("average_rating", 0))
    reviews = int(card.get("review_count", 0))
    positive = float(card.get("positive_share", 0))
    pros = safe_list(card.get("pros", []))
    cons = safe_list(card.get("cons", []))

    pros_text = ", ".join(pros[:3]) if pros else "its practical everyday use"
    cons_text = ", ".join(cons[:3]) if cons else "no repeated complaint clearly stood out"

    return (
        f"{name} has {reviews:,} reviews, averages {rating:.2f}/5, "
        f"and {positive:.1f}% of reviews are positive. Customers mainly mention "
        f"{pros_text} as strengths. The main complaints or issues are {cons_text}. "
        "Overall, it is worth comparing with similar products using the score, "
        "rating, review volume, and evidence strength shown above."
    )


def get_review_text(card):
    """Use the generated review, with a fallback for missing or broken text."""
    summary = str(card.get("summary", "") or "").strip()
    if len(summary) < 40 or "<div" in summary or "rr-stat-grid" in summary:
        return build_fallback_review(card)
    return summary


def evidence_class(value):
    return {
        "High": "rr-evidence-high",
        "Medium": "rr-evidence-medium",
        "Low": "rr-evidence-low",
    }.get(str(value), "")


def pick_label(rank):
    if int(rank) == 1:
        return "Top pick"
    if int(rank) == 2:
        return "Runner-up"
    if int(rank) == 3:
        return "Also consider"
    return f"Rank {int(rank)}"


def get_trust_score(card):
    """Use saved trust score, or calculate it for older generated JSON files."""
    if "trust_score" in card and pd.notna(card["trust_score"]):
        return float(card["trust_score"])

    rating_score = max(0, min(1, float(card.get("average_rating", 0)) / 5))
    positive_score = max(0, min(1, float(card.get("positive_share", 0)) / 100))
    risk_score = max(0, min(1, 1 - float(card.get("negative_share", 0)) / 100))

    # If old cards do not include normalized review volume, show a conservative fallback.
    volume_score = float(card.get("review_volume_score", 0.5))
    volume_score = max(0, min(1, volume_score))

    return round(40 * rating_score + 30 * volume_score + 20 * positive_score + 10 * risk_score, 1)


def evidence_tier(value):
    """Turn evidence labels into sortable numbers."""
    return {"High": 3, "Medium": 2, "Low": 1}.get(str(value), 0)


def clean_product_name(product_name):
    """Clean duplicated marketplace names for display and image matching."""
    text = str(product_name).replace("\r", "\n").strip()
    text = text.replace(",,,", "\n")
    text = text.replace("|", "\n")

    parts = []
    for part in text.splitlines():
        part = " ".join(part.split()).strip(" ,-\"")
        if part and part.lower() not in [item.lower() for item in parts]:
            parts.append(part)

    text = parts[0] if parts else str(product_name).strip()

    # Remove repeated marketplace brand prefixes.
    text = re.sub(r"(?i)^amazon\s*-\s*amazon\s+", "Amazon ", text)
    text = re.sub(r"(?i)^amazon\s+amazon\s+", "Amazon ", text)

    # Smooth common marketplace formatting while keeping the product identity.
    text = re.sub(r"(?i)^amazon\s*-\s*(kindle|fire|echo)", r"Amazon \1", text)
    text = re.sub(r"\s+", " ", text).strip(" ,-\"")
    text = text.replace(" Fire Hd ", " Fire HD ")
    text = text.replace(" Hd ", " HD ")

    return text


def normalize_image_match_name(value):
    """Normalize product names for safer image lookup."""
    text = clean_product_name(value).lower()
    text = re.sub(r"\b(includes|special|offers|with|without|packaging|may|vary)\b", " ", text)
    text = re.sub(r"\b(previous|generation|release|certified|refurbished|brand|new)\b", " ", text)
    text = re.sub(r"\b(color|free|shipping|powered|dolby|improved|sound|design|finish)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def extract_image_match_tokens(value):
    """Use meaningful product tokens for loose image matching."""
    blocked = {
        "amazon",
        "amazonbasics",
        "includes",
        "special",
        "offers",
        "with",
        "and",
        "the",
        "for",
        "black",
        "white",
        "new",
        "all",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(value).lower())
        if len(token) > 2 and token not in blocked
    ]


def is_usable_product_image_url(image_url):
    """Filter out barcodes and tiny thumbnails."""
    text = str(image_url).lower()
    if not text.startswith("http"):
        return False
    blocked = ["barcode", "upccode", "_ss40_", "_ss40", "._ss40", "_sl75_", "_sl160_"]
    return not any(term in text for term in blocked)


def score_image_url(image_url):
    """Prefer larger product images over thumbnails."""
    text = str(image_url).lower()
    score = 0

    if text.startswith("https://"):
        score += 5
    if "images-na.ssl-images-amazon.com" in text or "m.media-amazon.com" in text:
        score += 10
    if "bbystatic.com" in text or "bhphoto.com" in text:
        score += 8
    if "_sl" in text:
        score += 12
    if "_sx" in text or "_sy" in text or "_ux" in text or "_uy" in text:
        score += 6

    dimensions = [int(value) for value in re.findall(r"_(?:sl|sx|sy|ux|uy|ac_sl)(\d+)", text)]
    dimensions.extend(int(value) for value in re.findall(r"s-l(\d+)", text))
    if dimensions:
        score += min(max(dimensions), 2000) // 10

    if "ss40" in text or "thumb" in text:
        score -= 80

    return score


def best_image_url(value):
    """Return the best valid URL from a comma-separated image list."""
    candidates = []
    for raw_item in str(value).split(","):
        image_url = raw_item.strip()
        if is_usable_product_image_url(image_url):
            candidates.append((score_image_url(image_url), image_url))
    if not candidates:
        return ""
    return sorted(candidates, reverse=True)[0][1]


def add_image_aliases(lookup):
    """Add aliases copied from the original app."""
    for target_name, source_names in IMAGE_ALIAS_NAMES.items():
        target_key = normalize_image_match_name(target_name)
        if target_key in lookup:
            continue
        for source_name in source_names:
            source_key = normalize_image_match_name(source_name)
            image_url = lookup.get(source_key) or lookup.get(clean_product_name(source_name).lower())
            if image_url:
                lookup[target_key] = image_url
                break


@st.cache_data(ttl=600)
def load_product_image_lookup():
    """Load product image URLs from the old project/raw review files."""
    lookup = {}
    raw_dirs = [OLD_PROJECT_RAW_DIR, RAW_DIR]

    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            continue

        for file_path in sorted(raw_dir.glob("*.csv")):
            try:
                df = pd.read_csv(
                    file_path,
                    usecols=lambda col: col in ["name", "imageURLs"],
                    low_memory=False,
                )
            except ValueError:
                continue

            if "name" not in df.columns or "imageURLs" not in df.columns:
                continue

            df = df.dropna(subset=["name", "imageURLs"])
            for _, row in df.drop_duplicates("name").iterrows():
                image_url = best_image_url(row["imageURLs"])
                if not image_url:
                    continue
                name = clean_product_name(row["name"])
                lookup.setdefault(name.lower(), image_url)
                lookup.setdefault(normalize_image_match_name(name), image_url)

    add_image_aliases(lookup)
    return lookup


def is_confident_image_match(product_tokens, image_tokens, product_text, score):
    """Allow loose image matching only when product-family words agree."""
    if not product_tokens or not image_tokens:
        return False

    overlap_ratio = score / max(1, len(product_tokens))
    if score < 3 or overlap_ratio < 0.45:
        return False

    family_terms = [
        ["paperwhite"],
        ["voyage"],
        ["kindle"],
        ["fire", "tv"],
        ["fire", "stick"],
        ["fire", "tablet"],
        ["echo"],
        ["alexa"],
        ["battery"],
        ["batteries"],
        ["keyboard"],
        ["cable"],
        ["wire"],
        ["adapter"],
        ["folder"],
        ["litter"],
        ["kennel"],
        ["crate"],
    ]

    for terms in family_terms:
        if all(term in product_text for term in terms):
            return all(term in image_tokens for term in terms)

    return overlap_ratio >= 0.65


def get_product_image_url(image_lookup, row):
    """Match product images using exact and high-confidence loose matches."""
    product_name = row.get("name", "")
    category = row.get("category", "")
    candidates = [
        row.get("display_name", ""),
        clean_product_name(product_name),
        row.get("raw_product_name", ""),
        row.get("product_family", ""),
    ]

    for candidate in candidates:
        image_url = image_lookup.get(str(candidate).strip().lower())
        if image_url:
            return image_url
        image_url = image_lookup.get(normalize_image_match_name(candidate))
        if image_url:
            return image_url

    product_tokens = set(extract_image_match_tokens(product_name))
    product_text = f"{product_name} {category}".lower()
    if len(product_tokens) < 2:
        return ""

    best_url = ""
    best_score = 0
    best_name = ""
    for name, image_url in image_lookup.items():
        image_tokens = set(extract_image_match_tokens(name))
        if not image_tokens:
            continue
        score = len(product_tokens & image_tokens)
        if score > best_score:
            best_score = score
            best_url = image_url
            best_name = name

    image_tokens = set(extract_image_match_tokens(best_name))
    if is_confident_image_match(product_tokens, image_tokens, product_text, best_score):
        return best_url

    return ""


@st.cache_data(ttl=10)
def load_cards():
    if not CARDS_PATH.exists():
        st.error("Missing product cards. Run `python src/03_generation_baseline.py --sample-products 10` first.")
        st.stop()

    cards = json.loads(CARDS_PATH.read_text())
    frame = pd.DataFrame(cards)
    frame["display_name"] = frame["name"].apply(clean_product_name)
    frame["trust_score_for_app"] = frame.apply(get_trust_score, axis=1)
    frame["evidence_tier_for_app"] = frame["evidence_strength"].apply(evidence_tier)

    # Some raw product names clean to the same display name. Keep the strongest
    # evidence row so the app does not show duplicate product cards.
    frame = frame.sort_values(
        ["category", "display_name", "trust_score_for_app", "review_count"],
        ascending=[True, True, False, False],
    )
    frame = frame.drop_duplicates(subset=["category", "display_name"], keep="first").copy()

    image_lookup = load_product_image_lookup()
    frame["image_url"] = frame.apply(lambda row: get_product_image_url(image_lookup, row), axis=1)
    return frame


cards = load_cards()

st.markdown(
    """
    <div class="rr-hero">
      <div class="rr-hero-kicker">RoboReviews Intelligence</div>
      <div class="rr-hero-title">Shop smarter with review evidence</div>
      <div class="rr-hero-subtitle">
        A product recommendation dashboard powered by sentiment analysis, clustering,
        and human-style AI reviews.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="rr-sidebar-logo">
          <div class="rr-sidebar-logo-mark">
            <svg viewBox="0 0 180 130" role="img" aria-label="RoboReviews logo">
              <defs>
                <linearGradient id="rrProductGradient" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stop-color="#a3e635"/>
                  <stop offset="100%" stop-color="#0f766e"/>
                </linearGradient>
                <linearGradient id="rrLensGradient" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stop-color="#fff7ed"/>
                  <stop offset="100%" stop-color="#d7f7ef"/>
                </linearGradient>
              </defs>
              <rect x="34" y="34" width="84" height="70" rx="13" fill="url(#rrProductGradient)"/>
              <path d="M34 47 L50 34 L50 104 L34 93 Z" fill="#7cc63b"/>
              <path d="M50 46 C66 56 83 57 99 47" fill="none" stroke="#ecfdf5" stroke-width="8" stroke-linecap="round"/>
              <line x1="52" y1="69" x2="52" y2="101" stroke="#17212b" stroke-width="4"/>
              <line x1="60" y1="66" x2="60" y2="101" stroke="#17212b" stroke-width="3"/>
              <line x1="69" y1="70" x2="69" y2="101" stroke="#17212b" stroke-width="3"/>
              <line x1="78" y1="64" x2="78" y2="101" stroke="#17212b" stroke-width="4"/>
              <line x1="87" y1="70" x2="87" y2="101" stroke="#17212b" stroke-width="3"/>
              <line x1="96" y1="66" x2="96" y2="101" stroke="#17212b" stroke-width="3"/>
              <circle cx="92" cy="57" r="38" fill="url(#rrLensGradient)" stroke="#241d19" stroke-width="10"/>
              <line x1="120" y1="85" x2="153" y2="115" stroke="#241d19" stroke-width="14" stroke-linecap="round"/>
              <line x1="92" y1="36" x2="129" y2="8" stroke="#f59e0b" stroke-width="7" stroke-linecap="round"/>
              <line x1="96" y1="50" x2="139" y2="18" stroke="#f59e0b" stroke-width="6" stroke-linecap="round"/>
              <line x1="100" y1="64" x2="145" y2="31" stroke="#f59e0b" stroke-width="6" stroke-linecap="round"/>
              <path d="M111 74 l4 9 10 1 -8 6 2 10 -8-5 -9 5 3-10 -8-6 10-1z" fill="#f59e0b"/>
              <path d="M135 61 l4 9 10 1 -8 6 2 10 -8-5 -9 5 3-10 -8-6 10-1z" fill="#f59e0b"/>
            </svg>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Filters")

    categories = ["All categories"] + sorted(cards["category"].dropna().unique().tolist())
    selected_category = st.selectbox("Choose product category", categories)

    sort_option = st.selectbox(
        "Sort products by",
        [
            "Recommended ranking",
            "Highest rating",
            "Most reviews",
            "Lowest risk",
        ],
    )

    minimum_evidence = st.selectbox(
        "Evidence",
        [
            "Show all",
            "Medium + High only",
            "High only",
        ],
    )

    show_advanced_details = st.toggle("Show advanced details", value=False)

    st.markdown(
        """
        <div class="rr-sidebar-note">
          <strong>How to read this</strong><br>
          Product Score uses 40% rating, 30% review volume, 20% positive sentiment,
          and 10% risk control.<br><br>
          Evidence shows how much review data supports the score:<br>
          High = 100+ reviews<br>
          Medium = 30-99 reviews<br>
          Low = fewer than 30 reviews
        </div>
        """,
        unsafe_allow_html=True,
    )

filtered = cards.copy()
if selected_category != "All categories":
    filtered = filtered[filtered["category"] == selected_category].copy()

if minimum_evidence == "Medium + High only":
    filtered = filtered[filtered["evidence_strength"].isin(["Medium", "High"])]
elif minimum_evidence == "High only":
    filtered = filtered[filtered["evidence_strength"].eq("High")]

sort_config = {
    "Recommended ranking": (
        ["trust_score_for_app", "evidence_tier_for_app", "review_count"],
        [False, False, False],
    ),
    "Highest rating": (
        ["average_rating", "evidence_tier_for_app", "review_count"],
        [False, False, False],
    ),
    "Most reviews": (
        ["review_count", "evidence_tier_for_app", "trust_score_for_app"],
        [False, False, False],
    ),
    "Lowest risk": (
        ["negative_share", "evidence_tier_for_app", "trust_score_for_app"],
        [True, False, False],
    ),
}
sort_columns, sort_ascending = sort_config[sort_option]
filtered = filtered.sort_values(sort_columns, ascending=sort_ascending)

if filtered.empty:
    st.warning("No products match the selected evidence filter for this category.")
    st.stop()

max_products_to_show = max(1, len(filtered))
if max_products_to_show == 1:
    top_n = 1
    st.sidebar.caption("Number of products to show: 1")
else:
    top_n = st.sidebar.slider(
        "Number of products to show",
        min_value=1,
        max_value=max_products_to_show,
        value=max_products_to_show,
    )

filtered = filtered.head(top_n).copy()
filtered["rank"] = range(1, len(filtered) + 1)

st.markdown(
    f"""
    <div class="rr-section-title">
      <span class="rr-category-pill">{escape_text(selected_category)}</span>
      <h2>Reviewer-style comparison</h2>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Products are ordered by Product Score: 40% rating, 30% review volume, "
    "20% positive sentiment, and 10% risk control."
)

for row_start in range(0, len(filtered), 3):
    row_cards = filtered.iloc[row_start:row_start + 3]
    columns = st.columns(3)

    for column, (_, card) in zip(columns, row_cards.iterrows()):
        with column:
            pros = safe_list(card.get("pros", []))
            cons = safe_list(card.get("cons", []))
            evidence = card.get("evidence_strength", "High")

            with st.container(border=True):
                pro_html = "".join(f'<div class="rr-list-item rr-pro">{escape_text(item)}</div>' for item in pros)
                con_html = "".join(f'<div class="rr-list-item rr-con">{escape_text(item)}</div>' for item in cons)
                trust_score = get_trust_score(card)
                image_url = str(card.get("image_url", "")).strip()
                review_text = get_review_text(card)

                display_name = card.get("display_name", clean_product_name(card.get("name", "")))

                render_html(f"<div class='rr-pick-label'>{escape_text(pick_label(card['rank']))}</div>")
                if image_url and image_url.lower() != "nan":
                    st.image(image_url, width="stretch")
                else:
                    render_html(
                        """
                        <div class="rr-image-placeholder">
                            <div class="rr-image-placeholder-title">Image not confidently matched</div>
                            <div class="rr-image-placeholder-copy">
                                Showing no image is safer than showing the wrong product.
                            </div>
                        </div>
                        """
                    )

                render_html(
                    f"""
                    <span class="rr-category-pill">{escape_text(card["category"])}</span>
                    <div class="rr-card-title">{escape_text(display_name)}</div>

                    <div class="rr-stat-grid">
                        <div class="rr-stat rr-evidence-high">
                            <div class="rr-stat-label">Score</div>
                            <div class="rr-stat-value">{trust_score:.1f}/100</div>
                        </div>
                        <div class="rr-stat">
                            <div class="rr-stat-label">Rating</div>
                            <div class="rr-stat-value">{float(card["average_rating"]):.2f}/5</div>
                        </div>
                        <div class="rr-stat">
                            <div class="rr-stat-label">Reviews</div>
                            <div class="rr-stat-value">{int(card["review_count"]):,}</div>
                        </div>
                        <div class="rr-stat {evidence_class(evidence)}">
                            <div class="rr-stat-label">Evidence</div>
                            <div class="rr-stat-value">{escape_text(evidence)}</div>
                        </div>
                    </div>

                    <div class="rr-verdict">{escape_text(card.get("headline", ""))}</div>
                    <div class="rr-blog">{render_paragraphs(review_text)}</div>
                    """
                )

                with st.expander("Fit, strengths, and complaints / issues"):
                    render_html(
                        f"""
                        <div class="rr-card-subhead">Best fit</div>
                        <div class="rr-fit-box">{escape_text(card.get("best_for", ""))}</div>

                        <div class="rr-card-subhead">What customers like</div>
                        {pro_html or '<div class="rr-list-item rr-pro">No clear strength listed.</div>'}

                        <div class="rr-card-subhead">Complaints / issues</div>
                        {con_html or '<div class="rr-list-item rr-con">No clear concern listed.</div>'}
                        """
                    )

if EVIDENCE_PATH.exists():
    if show_advanced_details:
        with st.expander("Detailed ranking table"):
            display_columns = [
                "rank",
                "category",
                "name",
                "display_name",
                "trust_score_for_app",
                "evidence_strength",
                "average_rating",
                "review_count",
                "positive_share",
                "negative_share",
                "cluster_id",
                "product_family",
            ]
            available_columns = [column for column in display_columns if column in filtered.columns]
            st.dataframe(
                filtered[available_columns].rename(
                    columns={
                        "display_name": "clean_product_name",
                        "trust_score_for_app": "product_score",
                        "positive_share": "text_positive_rate",
                        "negative_share": "text_negative_rate",
                    }
                ),
                use_container_width=True,
            )

        with st.expander("Full product evidence table"):
            st.dataframe(pd.read_csv(EVIDENCE_PATH), use_container_width=True)
