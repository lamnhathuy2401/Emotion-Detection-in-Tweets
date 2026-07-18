# Emotion Detection in Tweets

**DS 423 — Machine Learning with Large Datasets**  
**Group 22 — NLP** | Duy Tan University, Da Nang

| Member | Student ID | Role |
|--------|------------|------|
| Lâm Nhật Huy | 29211465364 | Data & Analysis Lead |
| Nguyễn Thị Lam Giang | 29201458314 | Modeling & Evaluation Lead |

**Repository:** https://github.com/lamnhathuy2401/Emotion-Detection-in-Tweets

## Problem

Brands and researchers need to automatically detect emotions (joy, anger, sadness, fear, love, surprise) from short social-media text.
This project builds a **multi-class text classifier** that labels the emotion of each tweet.
The implementation is an offline notebook-based pipeline (not a deployed real-time system).

## Dataset

[Emotion Dataset (dair-ai)](https://huggingface.co/datasets/dair-ai/emotion) on Hugging Face.

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 16,000 | Training + cross-validation |
| Validation | 2,000 | Development check |
| Test | 2,000 | Final evaluation |

**6 classes:** sadness, joy, love, anger, fear, surprise (class-imbalanced; joy ≈ 33.5%, surprise ≈ 3.6%).

## Project Structure

```
Emotion-Detection-in-Tweets/
├── emotion_detection.ipynb   # Full runnable notebook (EDA → train → evaluate)
├── requirements.txt
├── README.md
├── report.tex                # LaTeX report (compile with XeLaTeX)
├── report.pdf                # Compiled report
├── Emotion Detection.pptx    # Presentation slides
└── outputs/                  # Figures and metrics from the notebook
    ├── eda_overview.png
    ├── confusion_matrix.png
    ├── f1_per_class.png
    ├── classification_report.txt
    └── model_comparison.csv
```

## Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
jupyter notebook emotion_detection.ipynb
```

## Approach

1. Load and explore the dataset (EDA)
2. Clean and preprocess tweet text (URLs, mentions, stopwords, lemmatization)
3. Vectorize with TF-IDF (unigrams + bigrams, max 10,000 features)
4. Compare baselines: Majority Class, Multinomial Naive Bayes, Linear SVM
5. Tune Logistic Regression with GridSearchCV (3-fold stratified CV)
6. Evaluate on the held-out test set: Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Results (Test Set)

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| Majority Class (always joy) | 34.75% | 0.1792 |
| TF-IDF + Multinomial Naive Bayes | 77.00% | 0.7345 |
| TF-IDF + Linear SVM | 90.05% | 0.9019 |
| **TF-IDF + Logistic Regression (tuned)** | **90.25%** | **0.9043** |

Best hyperparameters: `C=5.0`, `class_weight=balanced` (CV weighted F1 = 0.9009).

Charts and detailed metrics are saved under `outputs/`.

## Submission

- **Code:** `emotion_detection.ipynb` + `requirements.txt` + this README
- **Report:** `report.tex` and `report.pdf` (XeLaTeX)
- **Presentation:** `Emotion Detection.pptx` (problem, solution, results, contribution, APA references)
