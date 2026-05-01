import os
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect,
    url_for,
    session,
)
from markupsafe import Markup
from model import predict_image, predict_image_details
import utils
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image
import base64
import io
from auth import create_user, get_user_by_email, verify_password



app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = os.environ.get("AUTH_SECRET_KEY", "dev-secret-key-change-me")
BASE_DIR = Path(__file__).resolve().parent
supplement_info = pd.read_csv(BASE_DIR / 'supplement_info.csv', encoding='cp1252')
disease_info = pd.read_csv(BASE_DIR / 'disease_info.csv', encoding='cp1252')

PREDICTION_HISTORY = []
MAX_HISTORY = 1000


def _make_thumb_base64(img_bytes: bytes, size: int = 120) -> str:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")



@app.route('/', methods=['GET'])
def home():
    username = session.get("username") if session.get("user_id") else None
    return render_template("dashboard.html", username=username)


@app.route('/api/history', methods=['GET'])
def api_history():
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"history": PREDICTION_HISTORY})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' field"}), 400

    file = request.files["file"]
    img_bytes = file.read()

    details = predict_image_details(img_bytes, top_k=5)
    thumb_b64 = _make_thumb_base64(img_bytes)

    record = {
        "predicted_label": details["predicted_label"],
        "confidence": details["confidence"],
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "thumbnail_b64": thumb_b64,
        "lat": request.form.get("lat"),
        "lon": request.form.get("lon"),
        "region": request.form.get("region") or "Unknown Location",
    }

    PREDICTION_HISTORY.insert(0, record)
    del PREDICTION_HISTORY[MAX_HISTORY:]

    return jsonify({"details": details, "record": record})


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        try:
            user = get_user_by_email(email) if email else None
        except Exception:
            return render_template(
                "login.html",
                error="Could not connect to MongoDB. Please check your MongoDB setup.",
            )

        if not user or not verify_password(password, user["password_hash"]):
            return render_template("login.html", error="Invalid email or password.")

        session["user_id"] = str(user["_id"])
        session["email"] = user["email"]
        session["username"] = user["username"]
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match.")

        try:
            create_user(username=username, email=email, password=password)
        except ValueError as e:
            return render_template("signup.html", error=str(e))
        except Exception:
            return render_template(
                "signup.html",
                error="Could not connect to MongoDB. Please check your MongoDB setup.",
            )

        # After signup, log the user in.
        try:
            user = get_user_by_email(email)
        except Exception:
            return render_template(
                "signup.html",
                error="Could not connect to MongoDB after signup. Please try login again.",
            )
        if not user:
            return render_template("signup.html", error="Signup successful but login failed. Try login.")

        session["user_id"] = str(user["_id"])
        session["email"] = user["email"]
        session["username"] = user["username"]
        return redirect(url_for("home"))

    return render_template("signup.html")


@app.route('/logout', methods=['GET'])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            img = file.read()
            prediction = predict_image(img)
            print(prediction)
            res = Markup(utils.disease_dic[prediction])
            return render_template('display.html', status=200, result=res)
        except:
            pass
    return redirect(url_for('index.html'))


@app.route('/shop', methods=['GET', 'POST'])
def shop():
    return render_template('shop.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))



if __name__ == "__main__":
    app.run(debug=True)
