"""Export generated product reviews to a readable PDF.

Run:
    python src/04_export_reviews_pdf.py

Input:
    outputs/articles/product_card_summaries.json

Output:
    outputs/articles/RoboReviews_generated_reviews.pdf
"""

from pathlib import Path
import json
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
ARTICLES = ROOT / "outputs" / "articles"
CARDS_PATH = ARTICLES / "product_card_summaries.json"
PDF_PATH = ARTICLES / "RoboReviews_generated_reviews.pdf"


def wrap_text(text, width):
    """Wrap long text so it fits nicely in the PDF."""
    text = str(text or "").replace("\n", " ")
    return "\n".join(textwrap.wrap(text, width=width))


def load_cards():
    """Load generated product cards."""
    if not CARDS_PATH.exists():
        raise FileNotFoundError(
            "Missing outputs/articles/product_card_summaries.json. "
            "Run python src/03_generation_baseline.py first."
        )

    cards = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    cards = sorted(cards, key=lambda card: (card.get("category", ""), card.get("rank", 999)))
    return cards


def make_pdf(cards):
    """Create a multi-page PDF with category, product, and review text."""
    with PdfPages(PDF_PATH) as pdf:
        for index, card in enumerate(cards):
            fig, axis = plt.subplots(figsize=(11.7, 8.3))  # A4 landscape
            axis.axis("off")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)

            axis.text(
                0.03,
                0.97,
                "RoboReviews Generated Product Reviews",
                fontsize=18,
                fontweight="bold",
                ha="left",
                va="top",
            )
            axis.text(
                0.03,
                0.925,
                "Category, product, and AI-generated review text",
                fontsize=10,
                color="#475569",
                ha="left",
                va="top",
            )

            category = card.get("category", "")
            product = wrap_text(card.get("name", ""), 92)
            review = wrap_text(card.get("summary", ""), 110)

            # One large card per product keeps the long review readable.
            axis.add_patch(
                plt.Rectangle(
                    (0.03, 0.08),
                    0.94,
                    0.78,
                    facecolor="#ffffff",
                    edgecolor="#cbd5e1",
                    linewidth=0.9,
                )
            )

            axis.text(
                0.06,
                0.81,
                "Category",
                fontsize=8,
                color="#64748b",
                fontweight="bold",
                ha="left",
                va="top",
            )
            axis.text(
                0.06,
                0.775,
                category,
                fontsize=11,
                color="#0f5f59",
                fontweight="bold",
                ha="left",
                va="top",
            )

            axis.text(
                0.06,
                0.71,
                "Product",
                fontsize=8,
                color="#64748b",
                fontweight="bold",
                ha="left",
                va="top",
            )
            axis.text(
                0.06,
                0.675,
                product,
                fontsize=12,
                color="#111827",
                fontweight="bold",
                ha="left",
                va="top",
                linespacing=1.15,
            )

            axis.text(
                0.06,
                0.54,
                "Review text",
                fontsize=8,
                color="#64748b",
                fontweight="bold",
                ha="left",
                va="top",
            )
            axis.text(
                0.06,
                0.50,
                review,
                fontsize=10,
                color="#111827",
                ha="left",
                va="top",
                linespacing=1.35,
            )

            page_number = index + 1
            total_pages = len(cards)
            axis.text(
                0.98,
                0.02,
                f"Page {page_number} of {total_pages}",
                fontsize=8,
                color="#64748b",
                ha="right",
                va="bottom",
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main():
    cards = load_cards()
    make_pdf(cards)
    print(f"Saved PDF: {PDF_PATH}")
    print(f"Products included: {len(cards)}")


if __name__ == "__main__":
    main()
