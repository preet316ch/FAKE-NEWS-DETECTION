# 📰 FakeNewsGuard AI

A complete machine-learning web application for **fake news detection** using TF-IDF and Logistic Regression.

> **Important:** This project is an educational text-classification system. Its REAL/FAKE result reflects patterns learned from the training dataset and is not proof that a news claim is factually true or false.

## ✨ Features

- Fake/real news classification
- TF-IDF text representation
- Logistic Regression classifier
- Confidence score
- Flask web interface
- JSON prediction API
- Training and evaluation script
- Confusion matrix and classification report
- Clean GitHub-ready project structure

## 🧠 Architecture

```text
News Text
   ↓
Text Cleaning
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
REAL / FAKE + Confidence
   ↓
Flask Web UI / REST API
```

## 📁 Project Structure

```text
FakeNewsGuard-AI/
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── news_dataset.csv
├── model/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/script.js
├── notebooks/
│   └── fake_news_analysis.ipynb
└── screenshots/
```

## 🚀 Run Locally

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This creates:

```text
model/fake_news_model.pkl
model/tfidf_vectorizer.pkl
```

### 4. Start the web app

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 🔌 API

Endpoint:

```text
POST /api/predict
```

Example JSON:

```json
{
  "text": "A sample news article goes here."
}
```

Example response:

```json
{
  "label": "REAL",
  "confidence": 87.42,
  "message": "The model classified this article as likely real based on patterns learned from the training data."
}
```

## 📊 Dataset

The included `data/news_dataset.csv` is a **small demonstration dataset** so the repository works immediately.

For a serious academic project, replace it with a larger, properly sourced dataset such as a public fake-news dataset. Keep the same two columns:

```csv
text,label
"news article text",1
"another article",0
```

Where:

- `0` = FAKE
- `1` = REAL

After replacing the dataset, retrain:

```bash
python train_model.py
```

## 🎓 IBM Project / Viva Talking Points

### Why TF-IDF?
TF-IDF converts news text into numerical features while giving higher importance to words that are useful for distinguishing documents.

### Why Logistic Regression?
It is fast, interpretable, works well with sparse TF-IDF features, and provides probability estimates for confidence scoring.

### Why not rely only on accuracy?
A dataset can be imbalanced. Precision, recall, F1-score and the confusion matrix should also be checked.

### Limitations
The model can make mistakes with satire, new events, unfamiliar topics, biased datasets, copied articles, and adversarial wording. It does not independently verify facts against trusted sources.

## 🌐 GitHub Upload

```bash
git init
git add .
git commit -m "Initial commit - FakeNewsGuard AI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/FakeNewsGuard-AI.git
git push -u origin main
```

## 📜 License

MIT License.
