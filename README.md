# 🐦 Emotion Detection in Tweets

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vPYqttNY-EvITddUZ_FBuzNg78t6m5YN?usp=sharing)

> A supervised machine learning pipeline powered by **scikit-learn** and **NLTK** — designed to classify fine-grained emotions from short, noisy social media text.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Flowchart](#flowchart)
- [Project Structure](#project-structure)
- [Pipeline Explained](#pipeline-explained)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Results & Evaluation](#results--evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Credits](#credits)

---

## 🔍 Overview

**Emotion Detection in Tweets** is an AI-powered text classification system that processes informal, short-form social media text to detect underlying human emotions. Moving beyond standard binary positive/negative sentiment analysis, this project categorizes tweets into six distinct fine-grained emotions:

- **Joy** 
- **Sadness**
- **Anger**
- **Fear**
- **Love**
- **Surprise**

The system utilizes a deliberate, supervised machine learning architecture to strip away social media noise, translate words into mathematical features via TF-IDF, and classify the text using a highly-tuned Logistic Regression engine.

---

## ✨ Features

| Feature | Description |
|---|---|
| ☁️ **Cloud-Ready** | Instantly runnable via Google Colab with no local setup required. |
| 🧹 **Robust Preprocessing** | Strips URLs, mentions, and non-alphabetic noise while preserving hashtag keywords. |
| 🔤 **Lexical Normalization** | Uses NLTK for stopword removal and lemmatization to standardize vocabulary. |
| 🧮 **TF-IDF Vectorization** | Translates textual data into a sparse $16,000 \times 10,000$ matrix using unigrams and bigrams. |
| 🤖 **Machine Learning Backend** | Tuned Logistic Regression model utilizing balanced class weighting to handle minority classes (e.g., Surprise). |
| 📊 **Extensive Evaluation** | In-depth performance metrics tracking including Accuracy, Precision, Recall, F1-score, and Confusion Matrices. |

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     EMOTION DETECTION PIPELINE                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
         ┌────────────────────────▼──────────────────────────┐
         │                   INPUT LAYER                     │
         │   🐦 Raw Tweets (Hugging Face dair-ai Dataset)    │
         └────────────────────────┬──────────────────────────┘
                                  │
         ┌────────────────────────▼──────────────────────────┐
         │               PREPROCESSING LAYER                 │
         │   • Lowercasing & URL/Mention Removal             │
         │   • Non-alphabetic character stripping            │
         │   • NLTK Stopword Removal & Lemmatization         │
         └────────────────────────┬──────────────────────────┘
                                  │
         ┌────────────────────────▼──────────────────────────┐
         │               FEATURE ENGINEERING                 │
         │   • TF-IDF Vectorization (scikit-learn)           │
         │   • Max Features: 10,000 | N-grams: (1, 2)        │
         └────────────────────────┬──────────────────────────┘
                                  │
         ┌────────────────────────▼──────────────────────────┐
         │               CLASSIFICATION LAYER                │
         │   • Multi-class Logistic Regression               │
         │   • Tuned Hyperparameters: C=5.0, balanced        │
         └────────────────────────┬──────────────────────────┘
                                  │
         ┌────────────────────────▼──────────────────────────┐
         │                   OUTPUT LAYER                    │
         │   📊 6 Emotion Classes:                           │
         │   Joy | Sadness | Anger | Fear | Love | Surprise  │
         └───────────────────────────────────────────────────┘
```

---

## 🔄 Flowchart

### Data Preprocessing Flow

```text
                    ┌─────────────────────┐
                    │ RAW SOCIAL MEDIA    │
                    │ @user I am feeling  │
                    │ SO #excited today!  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ NORMALIZE & STRIP   │
                    │ Remove URLs, @ tags │
                    │ and lowercasing     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │       FILTER        │
                    │ Remove punctuation  │
                    │ and stopwords       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    LEMMATIZATION    │
                    │ ['feel', 'excit',   │
                    │  'today']           │
                    └─────────────────────┘
```

---

## 📁 Project Structure

```text
doan/
│
├── emotion_detection.ipynb        # Main notebook (Pipeline & Training)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── outputs/                       # Saved figures and reports
│
└── report.tex                     # LaTeX source for the project report
```

---

## 🧩 Pipeline Explained

### 1. Text Preprocessing
Social media text is notoriously noisy. The preprocessing module strips out non-informative elements (URLs, mentions, punctuation) while keeping informative features like hashtag keywords. Text is then lemmatized using **NLTK** to reduce inflected forms to their common base, shrinking the overall vocabulary.

### 2. Feature Engineering (TF-IDF)
Cleaned text is mapped into mathematical space using Term Frequency-Inverse Document Frequency. 
- **Max features:** 10,000 
- **N-gram range:** Unigrams and Bigrams (1, 2)
- **Min_df:** 2 (ignores extremely rare terms)

### 3. Model Selection & Tuning
Several baselines (Naive Bayes, Linear SVM) were tested. **Logistic Regression** was chosen for its optimal balance of speed, interpretability, and performance on high-dimensional sparse data. It was tuned via 3-fold stratified cross-validation (`GridSearchCV`) with a regularization strength of `C=5.0` and balanced class weights to penalize errors on minority classes.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| **scikit-learn** | TF-IDF Vectorization, Logistic Regression, GridSearchCV, Metrics |
| **NLTK** | Stopword removal, tokenization, and lemmatization |
| **Hugging Face Datasets** | Loading the `dair-ai` emotion dataset |
| **Pandas / NumPy** | Data manipulation and numerical operations |
| **Matplotlib / Seaborn** | Exploratory Data Analysis (EDA) and Result Visualizations |

---

## 📊 Dataset

The project utilizes the **Emotion Dataset (`dair-ai`)** hosted on Hugging Face.
- **Total Tweets:** 20,000
- **Train:** 16,000 (80%)
- **Validation:** 2,000 (10%)
- **Test:** 2,000 (10%)

*Note: The dataset exhibits natural class imbalance, with Joy (33.5%) and Sadness (29.2%) dominating, and Surprise (3.6%) as the distinct minority class.*

---

## 🏆 Results & Evaluation

The tuned Logistic Regression engine successfully captured the emotional signals, significantly outperforming the baseline approaches on the strictly held-out test set.

| Metric | Score |
|---|---|
| **Accuracy** | **90.25%** |
| **Weighted Precision** | 0.9091 |
| **Weighted Recall** | 0.9025 |
| **Weighted F1-Score** | **0.9043** |

*The model demonstrated a massive +55.5% accuracy improvement over a simple majority-class baseline.*

---

## ⚙️ Installation

*(Not required if using Google Colab)*

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook / JupyterLab

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/lamnhathuy2401/Emotion-Detection-in-Tweets.git
cd Emotion-Detection-in-Tweets

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts ctivate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### ☁️ Option 1: Run in Google Colab (Recommended)
You can run the entire pipeline directly in your browser without any local setup. Click the badge below to open the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vPYqttNY-EvITddUZ_FBuzNg78t6m5YN?usp=sharing)

### 💻 Option 2: Run Locally
To run the pipeline and replicate the results on your machine:

1. Launch Jupyter Notebook in the project directory:
   ```bash
   jupyter notebook
   ```
2. Open `emotion_detection.ipynb`.
3. Run the cells sequentially to load the dataset, execute preprocessing, train the Logistic Regression model, and generate the evaluation metrics and confusion matrices.

---

## 🔮 Future Work

- **Context-Aware Transformers:** Implement models like DistilBERT to capture semantics and accurately process negation (e.g., distinguishing "not happy").
- **Minority Class Augmentation:** Use data augmentation techniques to bolster underrepresented emotions like Surprise.
- **API Deployment:** Transition from an offline Jupyter notebook into a real-time REST API for live dashboard integration.

---

## 🤝 Credits

**Group 22 - NLP | Duy Tan University**
- Lam Nhat Huy (29211465364)
- Nguyen Thi Lam Giang (29201458314)

---

<p align="center">Built with 🐦 Hugging Face · 🧠 scikit-learn · 📚 NLTK</p>
