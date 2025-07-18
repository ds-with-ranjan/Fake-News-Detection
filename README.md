# 📰 Fake News Detection using Machine Learning & NLP

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Issues](https://img.shields.io/github/issues/ds-with-ranjan/Fake-News-Detection)](https://github.com/ds-with-ranjan/Fake-News-Detection/issues)
[![Stars](https://img.shields.io/github/stars/ds-with-ranjan/Fake-News-Detection?style=social)](https://github.com/ds-with-ranjan/Fake-News-Detection/stargazers)

> A powerful machine learning project that classifies news as **Real** or **Fake** using NLP and classification models.

---

## 👨‍💻 Author

**Ranjan Chakrabortty**  
📧 Email: [ranjanchakrabortty4@gmail.com](mailto:ranjanchakrabortty4@gmail.com)  
🔗 GitHub: [@ds-with-ranjan](https://github.com/ds-with-ranjan)

---

## 📌 Overview

In the modern digital era, fake news can spread quickly and influence public opinion. This project leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** to identify whether a news article is real or fake. It includes:

- Data preprocessing
- TF-IDF vectorization
- Logistic Regression & Passive Aggressive Classifier
- Model evaluation and prediction

---

## 🚀 Features

- 🧠 Machine Learning with Scikit-learn
- 🧹 Text cleaning and preprocessing with NLTK
- 📈 TF-IDF based feature extraction
- ✅ Accuracy > 93% with PassiveAggressiveClassifier
- 🖥️ Ready for GUI/Streamlit integration

---

## 📊 Demo

> *(Optional: Add demo image here)*  
![Demo](docs/demo.png)

---

## 📂 Dataset

- **Source**: [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- **Files**: `Fake.csv`, `True.csv`
- **Columns**:
  - `title`, `text`, `subject`, `label`

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/ds-with-ranjan/Fake-News-Detection
cd Fake-News-Detection

2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

3. Install dependencies
pip install -r requirements.txt

4. Train the model
python code/classifier.py

5. Predict using saved model
python code/predict.py

| Model                         | Accuracy | F1 Score |
| ----------------------------- | -------- | -------- |
| Logistic Regression           | 92.5%    | 92.0     |
| Passive Aggressive Classifier | 93.1%    | 93.4     |
| Multinomial Naive Bayes       | 89.7%    | 89.0     |

✅ Best Model: Passive Aggressive Classifier


🗂 Project Structure

Fake-News-Detection/
│
├── code/
│   ├── classifier.py          # Model training
│   ├── predict.py             # Prediction using saved model
│   ├── vectorization.py       # TF-IDF vectorizer
│   └── train_test_split.py    # Splitting the dataset
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── valid.csv
│
├── final_model.pkl            # Trained model
├── requirements.txt
├── LICENSE
└── README.md

🔮 Future Improvements
Use advanced models like LSTM or BERT

Real-time prediction from live news feeds

Web deployment using Flask or Streamlit

Multilingual detection support

📜 License
This project is licensed under the MIT License. See LICENSE for details.

🙌 Acknowledgements
Scikit-learn

NLTK

Kaggle Dataset


🌟 Show Your Support

---

Would you like me to:
- Upload this to your GitHub repo directly (if you add me as collaborator)?
- Help you add demo images or a Streamlit UI?

Let me know how you'd like to proceed!











 

