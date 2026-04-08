🧠 VeriNews AI – Explainable Fake News Detection System

🔗 Research Paper: https://ijrti.org/viewpaperforall.php?paper=IJRTI2603205

---

📌 Overview

VeriNews AI is an intelligent fake news detection and verification system that leverages Machine Learning and Explainable AI to classify news articles as Real, Fake, or Uncertain.

The system goes beyond simple classification by providing transparency through confidence scores, visual analytics, and explainable insights, making it a reliable decision-support tool.

---

✨ Key Features

- 🔍 Fake vs Real News Classification
- 📊 Confidence Score Generation
- 📈 Interactive Dashboard (Charts & Analytics)
- 🧠 SHAP-based Explainability (Influential Words)
- ⚠️ Sensational Score & Risk Recommendation
- 🔐 User Authentication (Login & Register)
- 🕘 History Tracking Dashboard
- 🌙 Dark Mode Interface

---

🛠️ Tech Stack

Frontend

- HTML
- Tailwind CSS
- JavaScript

Backend

- Flask (Python)

Machine Learning

- Logistic Regression
- TF-IDF Vectorization
- Naive Bayes
- BERT 

Explainability

- SHAP (SHapley Additive exPlanations)

Database

- SQLite

---

📂 Project Structure

fake-news-detector/
│
├── app.py
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── login.html
│   ├── register.html
│
├── static/
│   └── style.css
│
├── src/
│   ├── predict.py
│   ├── preprocess.py
│   ├── bert_model.py
│   ├── shap_explainer.py
│
├── model/
│   ├── tfidf.pkl
│   ├── naive_bayes.pkl

---

⚙️ Installation & Setup

git clone https://github.com/barsharajput/fake-news-detector.git
cd fake-news-detector

pip install -r requirements.txt

python app.py

Open your browser and visit:

http://127.0.0.1:5000

---

⚙️ How It Works

1. User inputs news text
2. Text preprocessing is performed
3. Selected ML model predicts the label
4. Confidence score is calculated
5. Results are visualized in dashboard
6. SHAP highlights important words influencing prediction

---

📄 Research Contribution

This project is backed by a published research paper:

👉 https://ijrti.org/viewpaperforall.php?paper=IJRTI2603205

---

⚠️ Disclaimer

This system is intended for educational and research purposes only.
Predictions are probabilistic and should not be considered as verified facts.

---

👩‍💻 Author

Barsha Singh
AI/ML Developer

---
