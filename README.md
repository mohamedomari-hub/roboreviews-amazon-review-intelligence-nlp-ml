# RoboReviews - Amazon Review Intelligence


<img width="713" height="339" alt="Demo_roboreview" src="https://github.com/user-attachments/assets/213d776b-ae1c-4a42-8036-6e3ea5e3b777" />


This is the Amazon review intelligence project.

The goal is to turn Amazon review data into a simple reviewer-style product recommendation app. The project combines:

- sentiment analysis
- product clustering
- taxonomy correction
- Qwen/Ollama text generation
- Streamlit product cards

## Project Structure

```text
roboreviews-amazon-review-intelligence-project/
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw/
├── models/
├── outputs/
│   ├── articles/
│   ├── clustering/
│   ├── errors/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── 01_sentiment_baseline.py
│   ├── 02_clustering_baseline.py
│   └── 03_generation_baseline.py
├── README.md
└── requirements.txt
```

## Dataset Description

The project uses public Amazon product review CSV files from Datafiniti-style Amazon review datasets.

The raw data is stored locally in:

```text
data/raw/
├── 1429_1.csv
├── Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv
└── Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
```

The raw CSV files are not pushed to GitHub because the local `data/raw/` folder is large, around `395 MB`.

### Raw File Sizes

| File | Rows | Columns |
|---|---:|---:|
| `1429_1.csv` | 34,660 | 21 |
| `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv` | 5,000 | 24 |
| `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv` | 28,332 | 24 |
| **Combined raw data** | **67,992** | **27** |

### Main Columns Used

The most important columns are:

| Column | Meaning | Used For |
|---|---|---|
| `name` | Product name | product grouping, app display, image matching |
| `categories` | Raw product category text | baseline clustering and taxonomy checks |
| `primaryCategories` | Higher-level product category where available | extra category context |
| `asins` | Amazon product identifier | product identity check |
| `imageURLs` | Product image URLs where available | Streamlit product images |
| `reviews.rating` | Star rating from 1 to 5 | weak sentiment label and product score |
| `reviews.title` | Review title | sentiment text input |
| `reviews.text` | Full review text | sentiment text input and baseline clustering |
| `reviews.numHelpful` | Helpful vote count | available metadata, not core model input |
| `reviews.doRecommend` | Whether reviewer recommends product, where available | available metadata, not core model input |
| `reviews.username` | Reviewer username | not used for modeling |

### Missing Values And Cleaning

After combining the raw files:

| Field | Missing Values | Notes |
|---|---:|---|
| `reviews.rating` | 33 | rows without rating cannot receive weak sentiment labels |
| `reviews.text` | 1 | rows without review text are removed for sentiment modeling |
| `reviews.title` | 19 | missing titles are allowed; the review text is still used |
| `name` | 6,760 | mostly from one raw file; rows can still be used for review-level sentiment but not always for product-level grouping |
| `imageURLs` | 34,660 | images are only available in some raw files |

For sentiment modeling, rows with no rating or no review text are removed.

After this cleanup:

```text
67,958 review rows remain
```

### Rating Distribution

The rating distribution is strongly positive:

| Rating | Reviews |
|---:|---:|
| 1 star | 1,438 |
| 2 stars | 1,072 |
| 3 stars | 2,902 |
| 4 stars | 15,397 |
| 5 stars | 47,149 |

This imbalance is why Macro F1 is more important than accuracy.

### Sentiment Labels Created From Ratings

The sentiment task uses rating as a weak label:

```text
1-2 stars = negative
3 stars   = neutral
4-5 stars = positive
```

Label counts after cleanup:

| Sentiment Label | Reviews |
|---|---:|
| positive | 62,546 |
| neutral | 2,902 |
| negative | 2,510 |

### Product-Level Data

The combined cleaned data contains:

```text
125 unique product names
91 unique ASIN values
111 unique raw category strings
```

For clustering and generation, the data is aggregated from review-level rows into product-level rows. The final product-level pipeline uses one row per product with:

- review count
- average rating
- predicted positive / neutral / negative review shares
- common strengths
- common complaints or issues
- corrected taxonomy category
- product score

## How To Run From Fresh

Run the scripts in this order:

```bash
python src/01_sentiment_baseline.py
python src/02_clustering_baseline.py
python src/03_generation_baseline.py
```

Then run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

For a fast generation test:

```bash
python src/03_generation_baseline.py --sample-products 3
```

To rebuild evidence from raw reviews:

```bash
python src/03_generation_baseline.py --rebuild-evidence
```

## 1. Sentiment Baseline

Script:

```bash
python src/01_sentiment_baseline.py
```

### Input

The model uses:

- review title
- review text

The model does **not** use rating as an input feature.

Rating is used only to create weak sentiment labels:

```text
1-2 stars = negative
3 stars   = neutral
4-5 stars = positive
```

This is called weak labeling because the rating is only an approximate sentiment label. Some reviews contradict their rating.

### Cleaning And Preprocessing Tested

The sentiment script compares:

- raw text
- basic cleaning
- stemming
- lemmatization
- stop-word removal
- TF-IDF vectorization

Bag-of-Words was removed from the final simplified pipeline because TF-IDF was easier to explain and worked better for the selected models.

### Models Tested

The baseline compares:

- Logistic Regression
- Linear SVM
- Naive Bayes
- Random Forest
- XGBoost

I also tried transformer-style sentiment models:

- MiniLM sentence embeddings with a classifier
- standard BERT-style sentiment modeling

These did not perform as well as the tuned TF-IDF + Linear SVM setup for this dataset. The likely reasons were:

- the labels are weak labels created from star ratings, not manually annotated sentiment labels
- the dataset is highly imbalanced toward positive reviews
- many neutral reviews are mixed or contradictory
- short n-gram phrases such as `not worth`, `easy to use`, and `battery life` were captured very well by TF-IDF

Because of this, the final sentiment pipeline uses the simpler and more explainable TF-IDF + Linear SVM approach.

### Validation Setup

The script uses:

- stratified train/test split
- same split for all models
- 3-fold cross-validation on the training set
- Macro F1 as the main metric
- RandomizedSearchCV for the best Linear SVM setups

The test set is only used after model selection.

### Why Macro F1 Matters

The review dataset is heavily imbalanced toward positive reviews.

Accuracy and weighted F1 can look very high because the model can do well on the majority positive class. Macro F1 is more useful because it gives equal importance to:

- negative
- neutral
- positive

This makes the neutral and negative performance visible.

### N-Gram Improvement

TF-IDF with n-grams improved the sentiment model because sentiment often depends on short phrases, not only single words.

Examples:

```text
easy to use
battery life
not worth
works great
too expensive
```

Unigrams see words separately. Bigrams and trigrams help the model understand these short expressions.

The tuned SVM search included:

```text
ngram_range: (1,1), (1,2), (1,3)
min_df: 1, 2, 3
max_features: 15000, 30000, 50000
sublinear_tf: True
SVM C: 0.1, 0.5, 1.0, 2.0
class_weight: None or balanced
```

### Rating-Text Contradictions

Some reviews do not match their rating.

Examples:

```text
high rating but negative text
low rating but positive text
neutral rating but positive or negative text
```

This explains why neutral reviews are difficult. A 3-star review is often mixed: part positive and part negative.

The script saves contradiction examples in:

```text
outputs/errors/rating_text_contradictions.csv
outputs/errors/rating_text_contradictions_examples.md
```

### Extra Sentiment Experiments

The script also includes:

- downsampling / undersampling of the dominant positive class
- rating-text discrepancy filtering

These experiments were kept for explanation, but the original training distribution was kept because the experiments did not improve the best Macro F1.

### Main Sentiment Outputs

```text
outputs/metrics/sentiment_baseline_results.csv
outputs/metrics/random_search_sentiment_results.csv
outputs/metrics/best_sentiment_baseline_metrics.json
outputs/metrics/top3_tuned_sentiment_results.csv
outputs/metrics/best_tuned_model_classification_report.md
outputs/figures/best_tuned_sentiment_confusion_matrix_readable.png
outputs/figures/sentiment_tuned_top3_cv_vs_test_macro_f1.png
outputs/errors/rating_text_contradictions.csv
outputs/errors/rating_text_contradictions_examples.md
models/best_sentiment_baseline_model.joblib
```

## 2. Clustering Baseline And Improvement

Script:

```bash
python src/02_clustering_baseline.py
```

The clustering script compares two scenarios.

### Scenario 1: Before Improvement

The baseline clustering text uses:

```text
product name + category + review title + review text
```

This is useful as a baseline, but it is noisy.

Reason: many different products share similar review words:

```text
great
bad
easy
broken
works
love
problem
```

These words describe customer opinion, not product identity. So tablets, speakers, accessories, and non-electronics can look similar if customers use similar review language.

### Scenario 2: After Canonical Preprocessing

The improved clustering removes review text and focuses on product identity.

The improved text is created from the product name only:

```text
product_family tokens + helper word tokens
```

No review text is used in the improved clustering scenario.

### Canonical Preprocessing Steps

Example product:

```text
Amazon - Amazon Tap Portable Bluetooth and Wi-Fi Speaker - Black,,,
Amazon - Amazon Tap Portable Bluetooth and Wi-Fi Speaker - Black,,,
```

Step 1: clean repeated product names:

```text
Amazon Tap Portable Bluetooth and Wi-Fi Speaker - Black
```

Step 2: infer one product family from the cleaned name:

```text
family_echo_alexa_speaker
```

Step 3: add helper tokens only from words that exist in the product name:

```text
word_tap
word_speaker
```

Step 4: build the text row used for TF-IDF:

```text
family_echo_alexa_speaker family_echo_alexa_speaker ...
family_echo_alexa_speaker word_tap word_speaker ...
```

The family token is repeated strongly so TF-IDF gives product identity more importance than small variant words.

### Why Repetition Was Used

TF-IDF converts text into numbers based on token frequency and token rarity.

Repeating important family tokens gives them more weight in the feature matrix.

For example, a battery product becomes strongly represented by:

```text
family_accessory_battery
word_battery
word_alkaline
word_aa
```

Then TF-IDF turns those tokens into numeric columns. Each product is one row in the feature matrix. Each token is one possible column.

### Vectorization And Clustering

The improved pipeline uses:

- lemmatization
- TF-IDF
- TruncatedSVD with 8 components
- normalization
- KMeans
- Hierarchical / Agglomerative clustering
- Self-Organizing Map (SOM) as an additional clustering check

SVD is used to compress the TF-IDF matrix and remove tiny noise. The main improvement came from better product identity features, not from SVD alone.

### Clustering Metrics

The script scans:

```text
k = 2 to 20
```

For each k it calculates:

- silhouette score
- elbow-style within-cluster distance

KMeans and Hierarchical clustering are the main methods compared. SOM was also tested as a supporting method. Its results helped confirm the same general product-family separation found by KMeans and Hierarchical clustering, but KMeans / Hierarchical were kept as the main reported models because they were easier to explain and plot clearly.

Recent clean run result:

```text
before_with_reviews KMeans:        k=18, silhouette=0.724
before_with_reviews Hierarchical:  k=17, silhouette=0.723
after_canonical KMeans:            k=13, silhouette=0.965
after_canonical Hierarchical:      k=10, silhouette=0.959
```

This shows that canonical preprocessing created a much cleaner clustering signal.

### Presentation Plots

The script also saves easier-to-explain fixed k plots for:

- KMeans k=6
- KMeans k=7
- Hierarchical k=6
- Hierarchical k=7

These are useful for slides because fewer clusters are easier to explain than the automatically selected higher-k solution.

### Taxonomy Correction

After clustering, simple keyword rules create readable product categories:

```text
Fire Tablets
Echo & Alexa Devices
Kindle E-Readers
Electronic Accessories
Fire TV & Streaming
Non-Electronics
```

This correction makes the final app easier to understand. It also fixes obvious cases where a product name clearly belongs to a known product family.

### PCA Plots

PCA is used only for visualization.

Important:

```text
Clustering is done on the TF-IDF/SVD feature matrix.
PCA is only a 2D or multi-panel plot to help us see the clusters.
```

### Main Clustering Outputs

```text
outputs/metrics/baseline_before_after_clustering_comparison.csv
outputs/metrics/baseline_all_kmeans_k_scan.csv
outputs/metrics/baseline_all_hierarchical_k_scan.csv
outputs/figures/before_with_reviews_method_silhouette_and_elbow.png
outputs/figures/after_canonical_no_reviews_method_silhouette_and_elbow.png
outputs/figures/after_canonical_no_reviews_selected_clustering_pca_2d.png
outputs/figures/after_canonical_no_reviews_selected_clustering_pca_pairs.png
outputs/figures/after_canonical_no_reviews_cluster_sizes.png
outputs/figures/after_canonical_no_reviews_kmeans_6_selected_clustering_pca_2d.png
outputs/figures/after_canonical_no_reviews_kmeans_7_selected_clustering_pca_2d.png
outputs/figures/after_canonical_no_reviews_hierarchical_6_selected_clustering_pca_2d.png
outputs/figures/after_canonical_no_reviews_hierarchical_7_selected_clustering_pca_2d.png
outputs/clustering/after_canonical_no_reviews_selected_product_clusters.csv
outputs/clustering/after_canonical_no_reviews_cluster_summary.csv
outputs/clustering/after_canonical_no_reviews_taxonomy_corrected_clusters.csv
outputs/clustering/taxonomy_corrections.csv
outputs/clustering/product_groups.md
```

## 3. Generation With Qwen / Ollama

Script:

```bash
python src/03_generation_baseline.py
```

Generation creates product-card summaries for the Streamlit app.

The LLM is used as a writing layer, not as the decision-maker.

The script first builds structured evidence, then gives that evidence to Qwen.

### Evidence Given To Qwen

For each product, Qwen receives:

- cleaned product name
- corrected product category
- product type
- review count
- average rating
- positive review share
- negative review share
- evidence strength: High / Medium / Low
- main strengths
- main concerns
- comparison notes inside the same category
- recommendation instruction

### Product Score

The app uses a simple product score:

```text
40% rating
30% review volume
20% positive sentiment
10% risk control
```

Risk control means the product has fewer predicted negative reviews.

### Evidence Thresholds

Evidence strength is based on review count:

```text
High   = 100+ reviews
Medium = 30-99 reviews
Low    = fewer than 30 reviews
```

### Qwen Model

Default local model:

```text
qwen2.5:3b
```

You can change it with:

```bash
python src/03_generation_baseline.py --llm-model qwen2.5:7b
```

### Qwen Parameters

The Ollama call uses:

```text
temperature = 0.65
top_p = 0.90
repeat_penalty = 1.12
num_predict = 580
timeout = 180 seconds
```

Why:

- `temperature=0.65` gives friendly wording without becoming too random.
- `top_p=0.90` keeps output natural but focused.
- `repeat_penalty=1.12` reduces repeated phrases.
- `num_predict=580` gives enough room for 3 paragraphs plus pros and cons.
- `timeout=180` allows local Ollama generation to finish.

### Prompt Structure

The prompt asks Qwen to return a fixed format:

```text
HEADLINE
SUMMARY
BEST_FOR
PROS
CONS
```

The summary is guided to be:

- exactly 3 paragraphs
- around 60-75 words per paragraph
- around 190-220 words total
- friendly and human
- grounded in the provided evidence

### Prompt Improvements

Early generation could be too generic or mention irrelevant issues.

We improved the prompt by adding:

- product type detection
- product-specific strengths and concerns
- filtering rules for batteries and accessories
- few-shot examples for product types
- strict output labels
- required evidence sentence
- rules against invented claims

Example problem fixed:

A battery review should not talk about:

```text
screen display issues
apps
setup problems
```

So the generation script filters strengths and concerns based on product type.

### Required Evidence Sentence

Each generated review must include the exact evidence sentence once:

```text
It has X reviews, averages Y/5, and Z% of reviews are positive.
```

This keeps the generated article tied to real metrics.

### Generation Outputs

```text
outputs/articles/product_generation_evidence.csv
outputs/articles/product_card_summaries.json
outputs/articles/product_card_summaries.md
outputs/articles/product_evidence_for_streamlit.csv
outputs/articles/prompts/
models/generation_lemmatization_tfidf_sentiment_model.joblib
```

## 4. Streamlit App

App:

```bash
streamlit run app/streamlit_app.py
```

The app shows:

- product category filter
- ranking filter
- evidence filter
- product score
- rating
- review count
- evidence level
- generated review article
- best fit
- pros
- cons
- product images where confidently matched

The product cards are ranked inside each category using the product score.

## Full Pipeline Overview

```text
Raw Amazon reviews
        |
        v
Sentiment model
title + review text -> negative / neutral / positive
        |
        v
Product evidence
rating, review count, sentiment shares, strengths, concerns
        |
        v
Clustering
product-name canonical tokens -> product groups
        |
        v
Taxonomy correction
readable product categories
        |
        v
Qwen generation
structured evidence -> reviewer-style summaries
        |
        v
Streamlit app
ranked product recommendation cards
```

## Important Presentation Points

- The final presentation was designed for a short project format: about 7 minutes for slides and about 3 minutes for the Streamlit app demo.
- Ratings are weak labels, not perfect truth.
- Neutral reviews are hard because many 3-star reviews are mixed.
- Macro F1 matters because the dataset is imbalanced toward positive reviews.
- N-grams improved sentiment because short phrases carry sentiment.
- MiniLM and standard BERT-style sentiment trials were tested, but the simpler TF-IDF + Linear SVM model performed better and was easier to explain.
- Review text was noisy for clustering, so canonical product identity improved separation.
- SOM was also tested for clustering and supported the product-family pattern seen in KMeans and Hierarchical clustering.
- Qwen was only the writing layer. The evidence came from sentiment, clustering, ratings, and review counts.
- The Streamlit app turns the pipeline into a usable product recommendation experience.
