# Amazon Review Sentiment Classification

Sentiment classifier for **real Amazon product reviews**, built with a baseline ladder and
an honest classical-vs-transformer comparison.

**Dataset:** Amazon subset of the UCI *Sentiment Labelled Sentences* corpus (Kotzias et al.,
KDD 2015) — 1,000 real Amazon reviews, balanced 500 positive / 500 negative. Bundled in `data/`.

## Results (held-out test set, 80/20 stratified split, `random_state=42`)

| Model | Accuracy | F1 |
|---|---|---|
| Majority class (floor) | 0.500 | 0.00 |
| Linear SVM (TF-IDF) | 0.750 | 0.75 |
| Logistic Regression (TF-IDF) | 0.765 | 0.76 |
| Naive Bayes (TF-IDF) | 0.770 | 0.78 |
| **DistilBERT (fine-tuned)** | *run on Colab* | *run on Colab* |

The Logistic Regression model is fully interpretable — its strongest positive words are
*excellent, love, easy, comfortable, best*; strongest negative are *poor, worst, disappointed,
horrible, waste*.

## Two stages

1. **Classical NLP** (runs anywhere): TF-IDF features + Naive Bayes / Logistic Regression /
   Linear SVM, plus confusion matrix, interpretability (top words), and error analysis.
2. **Transfer learning** (`Section 6`): fine-tuning DistilBERT. This cell is written for
   **Google Colab** (free GPU) because it downloads pretrained weights from Hugging Face.
   Set `RUN_DISTILBERT = True` there, run it, and record the real accuracy into Section 7.
   No transformer number is reported until you've measured it.

## The point of the project

On only 800 short training reviews, a TF-IDF linear model is a strong, instant, interpretable
baseline (~0.77). A pretrained transformer is *expected* to edge it — because its language
understanding comes from pretraining, not from these few examples — but at the cost of
interpretability and compute. The project **measures** that trade-off rather than assuming the
bigger model wins.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook amazon_review_sentiment.ipynb   # "Run All" runs the classical pipeline
```

Pure scikit-learn for the classical part — no GPU, no heavy dependencies. Keep `data/` next to
the notebook (it loads `data/amazon_cells_labelled.txt`).

## Citation

D. Kotzias, M. Denil, N. de Freitas, P. Smyth. *From Group to Individual Labels using Deep
Features.* KDD 2015. (Amazon reviews originally from McAuley & Leskovec, 2013.)
