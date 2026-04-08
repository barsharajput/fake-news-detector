import os
import sys

history = []

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, redirect, render_template, request
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)

import models.user_model
from database.db import db
from models.user_model import History, User
from src.predict import predict_news

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "app.db")
app.config["SECRET_KEY"] = "secret123"

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# 🔐 PROTECTED HOME ROUTE
@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    result = None

    # ✅ ALWAYS define this first
    user_history = []

    if request.method == "POST":
        news_text = request.form.get("news")
        model_choice = request.form.get("model")

        if news_text:
            result = predict_news(news_text, model_choice)
            result["news_text"] = news_text

            # ✅ SAVE HISTORY
            if current_user.is_authenticated:
                history = History(
                    text=news_text,
                    result=result["label"],
                    confidence=result["confidence"],
                    user_id=current_user.id,
                )
                db.session.add(history)
                db.session.commit()

    # ✅ ALWAYS FETCH HISTORY (outside POST)
    if current_user.is_authenticated:
        user_history = History.query.filter_by(user_id=current_user.id).all()

    return render_template("index.html", result=result, history=user_history)


# 📝 REGISTER
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()

        return redirect("/login")

    return render_template("register.html")


# 🔑 LOGIN
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user and user.password == password:
            login_user(user)
            return redirect("/")

    return render_template("login.html")


# 🚪 LOGOUT
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")


@app.route("/dashboard")
@login_required
def dashboard():
    user_history = History.query.filter_by(user_id=current_user.id).all()
    return render_template("dashboard.html", history=user_history)


@app.route("/delete/<int:id>")
@login_required
def delete_history(id):
    item = History.query.get(id)
    if item and item.user_id == current_user.id:
        db.session.delete(item)
        db.session.commit()
    return redirect("/dashboard")


@app.route("/clear")
@login_required
def clear_history():
    History.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return redirect("/dashboard")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
