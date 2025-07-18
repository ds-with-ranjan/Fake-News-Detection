# 📰 Fake News Detection with Machine Learning

![App Demo](https://raw.githubusercontent.com/ds-with-ranjan/Fake-News-Detection/main/docs/demo.png)

A machine learning-based solution to detect fake news articles using NLP techniques and classification models. This project includes preprocessing, feature engineering, model training, evaluation, and an optional Streamlit web app interface.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Project Highlights](#project-highlights)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [EDA](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Evaluation](#model-evaluation)
- [Streamlit App](#streamlit-web-app)
- [How to Run](#how-to-run-locally)
- [Future Scope](#future-enhancements)
- [Author](#author)

---

## 📖 Overview
With misinformation becoming widespread, this project aims to classify news articles as **Real** or **Fake** using machine learning. It demonstrates how to use NLP pipelines with classification algorithms for practical applications.

---

## 🌟 Project Highlights
- ✅ Cleaned & preprocessed text using NLP
- ✅ Used **TF-IDF** vectorization
- ✅ Trained with Logistic Regression and Passive Aggressive Classifier
- ✅ Achieved over **93% accuracy**
- ✅ Optional **Streamlit UI** for real-time predictions
- ✅ Docker support for deployment *(optional)*

---

## 💻 Technologies Used
- Python
- NLTK
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Streamlit
- Docker

---

## 📂 Dataset
**[Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)** from Kaggle.  
- ~21,000 news articles  
- Balanced dataset with labels `FAKE` and `REAL`

---

## 📊 Exploratory Data Analysis

- Word frequency visualizations
- Word clouds of fake vs real
- Article length distribution
- Class balance plots

*See `EDA.ipynb` for full analysis*

---

## 🧠 Model Training

### 🔧 Preprocessing
- Stopword removal
- Punctuation cleaning
- Lemmatization using NLTK
- TF-IDF vectorizer for feature extraction

### 📈 Models Used

| Model                         | Accuracy | F1 Score |
|------------------------------|----------|----------|
| Logistic Regression          | 92.3%    | 0.91     |
| Passive Aggressive Classifier| 93.4%    | 0.93     |

---

## ✅ Evaluation

![Confusion Matrix](https://raw.githubusercontent.com/ds-with-ranjan/Fake-News-Detection/main/docs/confusion_matrix.png)

Evaluation metrics include:
- Classification report
- Accuracy and F1-Score
- Confusion matrix

---

## 🌐 Streamlit Web App

An interactive interface built with Streamlit lets you paste news content and get instant predictions.

### ▶️ Run App Locally:
```bash
streamlit run app.py

Features:

Input article text

Predict real or fake

Displays model confidence

🧪 How to Run Locally
🔻 Clone the Repo
bash
Copy
Edit
git clone https://github.com/ds-with-ranjan/Fake-News-Detection.git
cd Fake-News-Detection
🛠 Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🚀 Train Model (Optional)
Open and run FakeNewsDetection.ipynb in Jupyter Notebook

🌍 Launch Streamlit App
bash
Copy
Edit
streamlit run app.py
🐳 Docker Support (Optional)
Dockerfile is included for deployment:

dockerfile
Copy
Edit
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
Build & Run:

bash
Copy
Edit
docker build -t fake-news-app .
docker run -p 8501:8501 fake-news-app
🔮 Future Enhancements
 Add Deep Learning (LSTM/BERT)

 Detect fake sources (URLs/domains)

 Add multilingual support

 Explainable AI using SHAP

👤 Author
Ranjan Chakrabortty
📧 ranjanchakrabortty4@gmail.com
🌐 GitHub

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgements
Fake and Real News Dataset on Kaggle

Tutorials by Analytics Vidhya and Medium

vbnet
Copy
Edit



---

### 🔁 Summary of Your To-Do:
| Task | Status |
|------|--------|
| ✅ Paste this updated README into your repo | 🔁 Pending |
| 📸 Upload `docs/demo.png` and `docs/confusion_matrix.png` | 🔁 Pending |
| 🚀 (Optional) Ask me for Streamlit UI or Docker help | ❓ Let me know |

Would you like me to **build the `app.py` (Streamlit)** for your fake news prediction UI next?
