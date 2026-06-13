# Amazon Review Sentiment Classification

A sentiment classifier for **real Amazon product reviews**, built as a baseline ladder and an **honest classical-vs-transformer comparison** — the project measures the accuracy-vs-interpretability trade-off instead of assuming the bigger model wins.

**Dataset:** the Amazon subset of the UCI *Sentiment Labelled Sentences* corpus (Kotzias et al., KDD 2015) — 1,000 real Amazon reviews, balanced 500 positive / 500 negative, ~10 words each. Bundled in `data/`, no download needed.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20linear-orange)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-DistilBERT-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Results

Held-out test set of 200 reviews (80/20 **stratified** split, `random_state=42`; train 800 = 400 pos / 400 neg, test 200 = 100 / 100).

| Model | Accuracy | F1 | Notes |
| --- | ---: | ---: | --- |
| Majority class (floor) | 0.500 | 0.00 | what "always guess one label" scores |
| Linear SVM (TF-IDF) | 0.750 | 0.75 | |
| Logistic Regression (TF-IDF) | 0.765 | 0.76 | fully interpretable |
| Naive Bayes (TF-IDF) | 0.770 | 0.78 | best classical |
| **DistilBERT (fine-tuned)** | **0.91** | **0.91** | transfer learning, on Colab GPU |

The fine-tuned transformer beats the best classical baseline by **~14 points of accuracy** — but at the cost of interpretability and a GPU. With only 200 test reviews, the margin of error is roughly ±4 points, which the notebook calls out rather than over-claiming.

**What the linear model learned** (TF-IDF + Logistic Regression, 683 features learned on train):
strongest positive words — *great, works, good, excellent, love, easy, nice, price, comfortable*;
strongest negative — *poor, bad, disappointed, worst, horrible, waste*. Sensible drivers, and inspectable in seconds — the thing you lose when you move to the transformer.

---

## The point of the project

On only 800 short training reviews, a TF-IDF linear model is a strong, instant, **interpretable** baseline (~0.77). A pretrained transformer is *expected* to edge it — because its language understanding comes from pretraining, not from these few examples — but at the cost of interpretability and compute. The project **measures** that trade-off (and its uncertainty) instead of assuming the bigger model automatically wins.

---

## What's inside

```
amazon_review_sentiment.ipynb        # the full pipeline — run this
data/
  └── amazon_cells_labelled.txt       # 1,000 labelled Amazon reviews (bundled)
requirements.txt
```

Notebook structure: **load → EDA → stratified split → classical baselines (TF-IDF + Naive Bayes / Logistic Regression / Linear SVM) → confusion matrix + top-word interpretability + error analysis → DistilBERT fine-tuning (Section 6) → results → takeaways.**

---

## Two stages

1. **Classical NLP — runs anywhere.** TF-IDF features (unigrams + bigrams, English stop words, `min_df=2`) feeding Naive Bayes / Logistic Regression / Linear SVM, plus a confusion matrix, top-word interpretability, and error analysis. Pure scikit-learn, no GPU, runs in seconds.
2. **Transfer learning — Section 6, on Google Colab.** Fine-tunes `distilbert-base-uncased` (4 epochs, learning rate 2e-5). The cell is guarded by a `RUN_DISTILBERT` flag and written for a free Colab GPU because it pulls pretrained weights from Hugging Face. Set `RUN_DISTILBERT = True`, run it, and record the measured accuracy / F1 into Section 7 — no transformer number is reported until it has actually been run.

---

## Getting started

```bash
git clone https://github.com/mannatrajsingh/Amazon-Review-Sentiment.git
cd Amazon-Review-Sentiment

pip install -r requirements.txt
jupyter notebook amazon_review_sentiment.ipynb   # "Run All" runs the classical pipeline
```

The classical pipeline needs only scikit-learn — no GPU, no heavy dependencies. Keep `data/` next to the notebook (it loads `data/amazon_cells_labelled.txt`).

### Run the DistilBERT stage on Google Colab

[Open the notebook in Colab »](https://colab.research.google.com/github/mannatrajsingh/Amazon-Review-Sentiment/blob/main/amazon_review_sentiment.ipynb)

Switch the runtime to a GPU, set `RUN_DISTILBERT = True` in Section 6, and run all.
