"""
app.py  –  Flask backend for the Health Risk Prediction System
"""

import os, io, base64, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from flask import Flask, render_template, request, jsonify, session
from flask import redirect, url_for

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "model")

# ── load model artefacts ──────────────────────────────────────────────────────
rf_model    = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
scaler      = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feat_cols   = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
le_risk     = joblib.load(os.path.join(MODEL_DIR, "le_risk.pkl"))
mappings    = joblib.load(os.path.join(MODEL_DIR, "mappings.pkl"))

# ── colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "lightest": "#EEFAF6",
    "light":    "#C8EFE1",
    "mid":      "#5CC9A8",
    "dark":     "#12352C",
}

# ── app ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "hrps-secret-key-2024"


# ── helpers ───────────────────────────────────────────────────────────────────
def _fig_to_b64(fig):
    """Convert a matplotlib figure to a base-64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def build_input_row(form_data):
    """
    Convert raw HTML form values into a 1-row DataFrame aligned to feat_cols.
    Returns (row_df, error_string_or_None).
    """
    errors = []

    # ── numeric fields ────────────────────────────────────────────────────────
    def get_float(key, lo, hi, label):
        try:
            v = float(form_data[key])
            if not (lo <= v <= hi):
                errors.append(f"{label} must be between {lo} and {hi}.")
            return v
        except (KeyError, ValueError):
            errors.append(f"{label} is required and must be a number.")
            return None

    age    = get_float("age",    18,  120, "Age")
    weight = get_float("weight", 20,  300, "Weight (kg)")
    height = get_float("height", 50,  250, "Height (cm)")
    sleep  = get_float("sleep",   0,   24, "Sleep hours")

    # BMI (auto-computed server-side from weight & height)
    bmi = None
    if weight and height:
        bmi = round(weight / ((height / 100) ** 2), 1)

    # ── categorical fields ───────────────────────────────────────────────────
    smoking_raw   = form_data.get("smoking", "").strip().lower()
    alcohol_raw   = form_data.get("alcohol",  "").strip().lower()
    married_raw   = form_data.get("married",  "").strip().lower()
    exercise_raw  = form_data.get("exercise", "").strip().lower()
    profession_raw = form_data.get("profession", "").strip().lower()

    valid_smoking   = list(mappings["smoking"].keys())
    valid_alcohol   = list(mappings["alcohol"].keys())
    valid_married   = list(mappings["married"].keys())
    valid_exercise  = mappings["exercise_levels"]
    valid_profession = mappings["profession_list"]

    if smoking_raw   not in valid_smoking:
        errors.append(f"Smoking must be one of: {', '.join(valid_smoking)}.")
    if alcohol_raw   not in valid_alcohol:
        errors.append(f"Alcohol must be one of: {', '.join(valid_alcohol)}.")
    if married_raw   not in valid_married:
        errors.append(f"Marital status must be one of: {', '.join(valid_married)}.")
    if exercise_raw  not in valid_exercise:
        errors.append(f"Exercise level must be one of: {', '.join(valid_exercise)}.")
    if profession_raw not in valid_profession:
        errors.append(f"Profession must be one of: {', '.join(valid_profession)}.")

    if errors:
        return None, errors

    smoking_enc = mappings["smoking"][smoking_raw]
    alcohol_enc = mappings["alcohol"][alcohol_raw]
    married_enc = mappings["married"][married_raw]

    # BMI-sleep interaction
    bmi_sleep = bmi * sleep

    # ── build base dict ───────────────────────────────────────────────────────
    row = {
        "age":                   age,
        "weight":                weight,
        "height":                height,
        "sleep":                 sleep,
        "bmi":                   bmi,
        "smoking_encoded":       smoking_enc,
        "alcohol_encoded":       alcohol_enc,
        "married_encoded":       married_enc,
        "bmi_sleep_interaction": bmi_sleep,
    }

    # one-hot exercise
    for col in mappings["exercise_columns"]:
        level = col.replace("exercise_", "")
        row[col] = 1 if exercise_raw == level else 0

    # one-hot profession
    for col in mappings["profession_columns"]:
        prof = col.replace("profession_", "")
        row[col] = 1 if profession_raw == prof else 0

    # ── align to feat_cols (fill any missing with 0) ──────────────────────────
    row_df = pd.DataFrame([row])
    for c in feat_cols:
        if c not in row_df.columns:
            row_df[c] = 0
    row_df = row_df[feat_cols]

    return row_df, None


def generate_suggestions(form_data, prediction, probability):
    """Return a list of suggestion dicts based only on modifiable factors."""
    sugg = []

    bmi = None
    try:
        w = float(form_data.get("weight", 0))
        h = float(form_data.get("height", 1))
        bmi = w / ((h / 100) ** 2)
    except Exception:
        pass

    sleep  = float(form_data.get("sleep",  7))
    smoking = form_data.get("smoking", "no").lower()
    alcohol = form_data.get("alcohol", "no").lower()
    exercise = form_data.get("exercise", "medium").lower()

    # BMI / Weight
    if bmi:
        if bmi < 18.5:
            sugg.append({"icon": "scale", "title": "Healthy Weight",
                         "body": "Your BMI suggests you are underweight. Consider a calorie-rich, "
                                 "nutrient-dense diet and consult a nutritionist."})
        elif bmi >= 25 and bmi < 30:
            sugg.append({"icon": "scale", "title": "Weight Management",
                         "body": "A BMI in the overweight range increases cardiovascular risk. "
                                 "Aim for a 5-10 % reduction through balanced diet and exercise."})
        elif bmi >= 30:
            sugg.append({"icon": "scale", "title": "Weight Reduction Priority",
                         "body": "Obesity (BMI ≥ 30) is a major risk factor. A medically supervised "
                                 "weight-loss programme could significantly lower your health risk."})

    # Sleep
    if sleep < 6:
        sugg.append({"icon": "moon", "title": "Improve Sleep Duration",
                     "body": "Less than 6 hours of sleep is linked to elevated cardiovascular and "
                              "metabolic risk. Target 7-9 hours per night by maintaining a regular schedule."})
    elif sleep > 9:
        sugg.append({"icon": "moon", "title": "Review Sleep Quality",
                     "body": "Consistently sleeping more than 9 hours can indicate underlying health "
                              "issues. Consult a physician if you still feel fatigued."})

    # Smoking
    if smoking == "yes":
        sugg.append({"icon": "cigarette-off", "title": "Quit Smoking",
                     "body": "Smoking is one of the strongest modifiable risk factors. "
                              "Cessation programmes, nicotine replacement therapy, and GP support "
                              "can reduce risk substantially within months."})

    # Alcohol
    if alcohol == "yes":
        sugg.append({"icon": "wine-off", "title": "Moderate Alcohol Intake",
                     "body": "Regular alcohol consumption raises blood pressure and liver stress. "
                              "Following recommended limits (no more than 14 units/week) or abstaining "
                              "is advised for better health outcomes."})

    # Exercise
    if exercise in ("none", "low"):
        sugg.append({"icon": "activity", "title": "Increase Physical Activity",
                     "body": "Sedentary lifestyle significantly contributes to health risk. "
                              "Aim for at least 150 minutes of moderate aerobic activity per week "
                              "combined with 2 days of strength training."})
    elif exercise == "medium":
        sugg.append({"icon": "activity", "title": "Elevate Exercise Intensity",
                     "body": "You are moderately active. Progressing to higher-intensity workouts "
                              "(e.g., HIIT, brisk running) can further reduce risk."})

    # If low risk and no major issues, add a positive note
    if prediction == "low" and not sugg:
        sugg.append({"icon": "check-circle", "title": "Keep Up the Good Work",
                     "body": "Your current lifestyle factors appear healthy. Maintain your routine, "
                              "schedule regular health check-ups, and stay consistent."})

    return sugg


def generate_charts(form_data, bmi, probability):
    """Return dict of base-64 chart images."""
    charts = {}
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ── Chart 1: Risk probability gauge (horizontal bar) ─────────────────────
    fig, ax = plt.subplots(figsize=(6, 1.6), facecolor=PALETTE["lightest"])
    ax.set_facecolor(PALETTE["lightest"])
    pct = probability * 100
    ax.barh([0], [100], color=PALETTE["light"],  height=0.5, left=0)
    ax.barh([0], [pct],  color=PALETTE["mid"],    height=0.5, left=0)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Risk Probability (%)", color=PALETTE["dark"], fontsize=9)
    ax.tick_params(colors=PALETTE["dark"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["dark"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["light"])
    ax.text(pct + 1, 0, f"{pct:.1f}%", va="center",
            color=PALETTE["dark"], fontsize=9, fontweight="bold")
    ax.set_title("High-Risk Probability", color=PALETTE["dark"],
                 fontsize=10, fontweight="bold", pad=8)
    charts["gauge"] = _fig_to_b64(fig)

    # ── Chart 2: BMI category bar ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3), facecolor=PALETTE["lightest"])
    ax.set_facecolor(PALETTE["lightest"])
    categories = ["Underweight\n(<18.5)", "Normal\n(18.5-24.9)",
                  "Overweight\n(25-29.9)", "Obese\n(≥30)"]
    bar_colors = [PALETTE["light"], PALETTE["mid"], "#f0a500", "#e05050"]
    bmi_cat_idx = 0
    if bmi < 18.5:       bmi_cat_idx = 0
    elif bmi < 25:       bmi_cat_idx = 1
    elif bmi < 30:       bmi_cat_idx = 2
    else:                bmi_cat_idx = 3
    # Draw each bar individually so we can set per-bar alpha
    for idx, (cat, col) in enumerate(zip(categories, bar_colors)):
        alpha = 1.0 if idx == bmi_cat_idx else 0.3
        ax.bar(cat, 1, color=col, alpha=alpha, width=0.6)
        if idx == bmi_cat_idx:
            ax.text(idx, 0.5, "You", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
    ax.set_yticks([])
    ax.set_title(f"Your BMI: {bmi:.1f}", color=PALETTE["dark"],
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors=PALETTE["dark"], labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["light"])
    charts["bmi"] = _fig_to_b64(fig)

    # ── Chart 3: Lifestyle factor radar / bar ────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.2), facecolor=PALETTE["lightest"])
    ax.set_facecolor(PALETTE["lightest"])

    factors = ["Exercise", "Sleep", "Non-Smoker", "No Alcohol"]
    exercise_score = {"high": 1.0, "medium": 0.65, "low": 0.35, "none": 0.0}.get(
        form_data.get("exercise", "medium").lower(), 0.5)
    sleep_val = float(form_data.get("sleep", 7))
    sleep_score = min(max((sleep_val - 3) / 6, 0), 1)
    smoke_score  = 1.0 if form_data.get("smoking", "yes").lower() == "no" else 0.0
    alcohol_score = 1.0 if form_data.get("alcohol",  "yes").lower() == "no" else 0.0

    scores = [exercise_score, sleep_score, smoke_score, alcohol_score]
    bar_cols = [PALETTE["mid"] if s >= 0.6 else "#e07070" for s in scores]

    bars = ax.barh(factors, scores, color=bar_cols, height=0.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score (0 = poor, 1 = optimal)", color=PALETTE["dark"], fontsize=8)
    ax.set_title("Lifestyle Factor Scores", color=PALETTE["dark"],
                 fontsize=10, fontweight="bold")
    ax.tick_params(colors=PALETTE["dark"], labelsize=8)
    for bar, score in zip(bars, scores):
        ax.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", color=PALETTE["dark"], fontsize=8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["light"])
    charts["lifestyle"] = _fig_to_b64(fig)

    # ── Chart 4: Key numeric metrics vs healthy ranges ────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3), facecolor=PALETTE["lightest"])
    ax.set_facecolor(PALETTE["lightest"])
    metrics      = ["BMI", "Sleep (h)", "Age"]
    user_vals    = [bmi, float(form_data.get("sleep", 7)),
                    float(form_data.get("age", 30))]
    healthy_hi   = [24.9, 9.0, 60]
    healthy_lo   = [18.5, 7.0, 18]

    x = np.arange(len(metrics))
    width = 0.35
    # normalise to 0-1 using [lo, hi]
    norm_user = [min(max((v - lo) / max(hi - lo, 1), 0), 1.3)
                 for v, lo, hi in zip(user_vals, healthy_lo, healthy_hi)]
    norm_ideal = [1.0] * len(metrics)

    ax.bar(x - width / 2, norm_ideal, width, label="Ideal Range",
           color=PALETTE["mid"],   alpha=0.5)
    ax.bar(x + width / 2, norm_user,  width, label="Your Value",
           color=PALETTE["dark"],  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color=PALETTE["dark"], fontsize=8)
    ax.set_yticks([])
    ax.set_title("Your Metrics vs Ideal Range", color=PALETTE["dark"],
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, framealpha=0)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["light"])
    charts["metrics"] = _fig_to_b64(fig)

    return charts


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/form")
def form_page():
    professions   = mappings["profession_list"]
    exercise_lvls = mappings["exercise_levels"]
    return render_template("form.html",
                           professions=professions,
                           exercise_levels=exercise_lvls)


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    # Required name fields
    first_name = form_data.get("first_name", "").strip()
    last_name  = form_data.get("last_name",  "").strip()
    errors = []
    if not first_name:
        errors.append("First name is required.")
    if not last_name:
        errors.append("Last name is required.")

    row_df, field_errors = build_input_row(form_data)
    if field_errors:
        errors.extend(field_errors)

    if errors:
        return render_template("form.html",
                               professions=mappings["profession_list"],
                               exercise_levels=mappings["exercise_levels"],
                               errors=errors,
                               form_data=form_data)

    # ── inference ─────────────────────────────────────────────────────────────
    row_scaled  = scaler.transform(row_df)
    pred_enc    = rf_model.predict(row_scaled)[0]
    prediction  = le_risk.inverse_transform([pred_enc])[0]  # "high" / "low"
    proba       = rf_model.predict_proba(row_scaled)[0]
    # probability of HIGH risk — find correct class index dynamically
    classes_list = list(le_risk.classes_)
    high_idx     = classes_list.index("high") if "high" in classes_list else 0
    probability  = float(proba[high_idx])

    # ── derived values ────────────────────────────────────────────────────────
    weight = float(form_data.get("weight", 70))
    height = float(form_data.get("height", 170))
    bmi    = round(weight / ((height / 100) ** 2), 1)

    suggestions = generate_suggestions(form_data, prediction, probability)
    charts      = generate_charts(form_data, bmi, probability)

    from datetime import datetime
    return render_template(
        "results.html",
        first_name  = first_name,
        last_name   = last_name,
        prediction  = prediction,
        probability = round(probability * 100, 1),
        bmi         = bmi,
        suggestions = suggestions,
        charts      = charts,
        form_data   = form_data,
        now         = datetime.now().strftime("%d %B %Y, %H:%M"),
    )


@app.route("/reset")
def reset():
    return redirect(url_for("welcome"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
