import os
import sys

import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from deep_translator import GoogleTranslator
from flask import Flask, redirect, render_template, request, url_for
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)

from database.db import db
from models.user_model import History, User
from src.predict import predict_news

app = Flask(__name__)

# ==============================
# ✅ CONFIG
# ==============================
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "app.db")
app.config["SECRET_KEY"] = "secret123"

db.init_app(app)

# ==============================
# ✅ LOGIN MANAGER
# ==============================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"  # redirect if not logged in


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def extract_news_from_url(url):
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join([p.text for p in paragraphs])

        return text.strip()
    except:
        return None


# ==============================
# 🌐 LANDING PAGE (PUBLIC)
# ==============================
@app.route("/")
def landing():
    return render_template("landing.html")


# ==============================
# 🔐 MAIN APP (Protected)
# ==============================
@app.route("/home", methods=["GET", "POST"])
@login_required
def home():
    result = None
    user_history = History.query.filter_by(user_id=current_user.id).all()

    if request.method == "POST":
        news_text = request.form.get("news")
        url = request.form.get("url")
        model_choice = request.form.get("model")

        print("URL:", url)
        print("TEXT BEFORE:", news_text)

        # 🔥 URL → extract text
        if url and url.strip():
            try:
                headers = {"User-Agent": "Mozilla/5.0"}

                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")

                paragraphs = soup.find_all("p")
                news_text = " ".join([p.get_text() for p in paragraphs])

                news_text = news_text.replace("\n", " ").strip()

                print("TEXT AFTER:", news_text[:200])

            except Exception as e:
                print("ERROR:", e)
                return render_template(
                    "index.html",
                    error="⚠️ Failed to fetch article",
                    history=user_history,
                )

        # ❌ empty input
        if not news_text or news_text.strip() == "":
            return render_template(
                "index.html",
                error="⚠️ Enter URL or news text",
                history=user_history,
            )

        # 🌍 LANGUAGE DETECTION + TRANSLATION
        is_translated = False

        try:
            is_translated = False

            try:
                translated_text = GoogleTranslator(
                    source="auto", target="en"
                ).translate(news_text)

                # check if translation actually changed text
                if translated_text and translated_text != news_text:
                    news_text = translated_text
                    is_translated = True

            except Exception as e:
                print("Translation Error:", e)
                is_translated = True

                print("Translated Text:", news_text[:200])

        except Exception as e:
            print("Translation Error:", e)

        # 🔥 BERT FIX (after translation)
        if model_choice == "bert":
            news_text = news_text[:512]

        # ✅ prediction
        result = predict_news(news_text, model_choice)
        result["news_text"] = news_text

        language = "en"  # default
        is_translated = False

        try:
            translated_text = GoogleTranslator(source="auto", target="en").translate(
                news_text
            )

            # detect language by comparing
            if translated_text and translated_text != news_text:
                language = "hi"
                news_text = translated_text
                is_translated = True

        except Exception as e:
            print("Translation Error:", e)

        # ✅ pass to frontend
        result["translated"] = is_translated
        result["language"] = language

        history_item = History(
            text=news_text[:500],
            result=result["label"],
            confidence=result["confidence"],
            user_id=current_user.id,
        )

        db.session.add(history_item)
        db.session.commit()

    return render_template("index.html", result=result, history=user_history)


# ==============================
# 📝 REGISTER
# ==============================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # ❌ prevent duplicate users
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template("register.html", error="User already exists")

        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")


# ==============================
# 🔑 LOGIN
# ==============================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()

        if user and user.password == password:
            login_user(user)
            return redirect(url_for("home"))  # 🔥 goes to /home now
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


# ==============================
# 🚪 LOGOUT
# ==============================
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("landing"))  # back to landing


# ==============================
# 📊 DASHBOARD
# ==============================
@app.route("/dashboard")
@login_required
def dashboard():
    user_history = History.query.filter_by(user_id=current_user.id).all()
    return render_template("dashboard.html", history=user_history)


# ==============================
# 🗑 DELETE HISTORY
# ==============================
@app.route("/delete/<int:id>")
@login_required
def delete_history(id):
    item = History.query.get(id)

    if item and item.user_id == current_user.id:
        db.session.delete(item)
        db.session.commit()

    return redirect(url_for("dashboard"))


# ==============================
# 🧹 CLEAR HISTORY
# ==============================
@app.route("/clear")
@login_required
def clear_history():
    History.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return redirect(url_for("dashboard"))


# ==============================
# 🚀 RUN APP
# ==============================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(host="0.0.0.0", port=5000, debug=True)
