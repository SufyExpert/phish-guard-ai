"""
phishing_app.py  --  Flask backend for AI-Based Email Phishing Detection System
Pipeline: Email Text -> TF-IDF Vectorization -> Logistic Regression
Color palette: #FFFFFF | #FBAC41 | #FF6632 | #070033
"""

import os, io, base64, re, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime

# Google OAuth & Gmail API
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ── Load model artifacts ───────────────────────────────────────────────────────
pipeline = joblib.load(os.path.join(MODEL_DIR, "phishing_pipeline.pkl"))
metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "white":   "#FFFFFF",
    "amber":   "#FBAC41",
    "orange":  "#FF6632",
    "navy":    "#070033",
    "navy_mid":"#0D0050",
}

app = Flask(__name__)
app.secret_key = "epds-secret-key-2025"


# ── Helpers ────────────────────────────────────────────────────────────────────
def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_risk_signals(text: str) -> dict:
    """Extract rule-based signal indicators for display."""
    signals = {}
    raw = text.lower()

    # URL presence
    url_count = len(re.findall(r"http\S+|www\.\S+|\.com|\.net|\.org|\.info|\.xyz", raw))
    signals["urls"] = min(url_count, 10)

    # Urgency keywords
    urgency_words = ["urgent", "immediately", "act now", "limited time", "expires",
                     "suspended", "verify", "confirm", "click here", "account",
                     "password", "login", "security", "alert", "warning"]
    signals["urgency"] = sum(1 for w in urgency_words if w in raw)

    # Suspicious patterns
    suspicious = ["dear customer", "dear user", "winner", "congratulations",
                  "prize", "lottery", "free", "earn money", "make money",
                  "bank", "paypal", "credit card", "ssn", "social security"]
    signals["suspicious"] = sum(1 for w in suspicious if w in raw)

    # All-caps ratio
    words = text.split()
    caps = sum(1 for w in words if w.isupper() and len(w) > 2)
    signals["caps_ratio"] = round(caps / max(len(words), 1), 2)

    # Email addresses
    signals["email_addrs"] = len(re.findall(r"\S+@\S+\.\S+", text))

    # Grammar/special chars
    signals["special_chars"] = len(re.findall(r"[!$%&*]", text))

    return signals


def compute_threat_level(probability: float) -> dict:
    """Map phishing probability to threat level metadata."""
    if probability >= 0.85:
        return {"level": "Critical", "color": "#FF6632", "bg": "rgba(255,102,50,0.12)", "border": "rgba(255,102,50,0.40)",
                "icon": "alert-octagon", "description": "This email exhibits strong phishing signatures."}
    elif probability >= 0.65:
        return {"level": "High", "color": "#FF6632", "bg": "rgba(255,102,50,0.08)", "border": "rgba(255,102,50,0.25)",
                "icon": "alert-triangle", "description": "This email shows multiple phishing indicators."}
    elif probability >= 0.40:
        return {"level": "Medium", "color": "#FBAC41", "bg": "rgba(251,172,65,0.10)", "border": "rgba(251,172,65,0.30)",
                "icon": "alert-circle", "description": "Moderate risk — proceed with caution."}
    else:
        return {"level": "Low", "color": "#22c55e", "bg": "rgba(34,197,94,0.10)", "border": "rgba(34,197,94,0.25)",
                "icon": "check-circle", "description": "This email appears to be safe."}


def adjust_threat_with_rules(sender: str, subject: str, body: str, raw_prob: float) -> tuple[float, int]:
    """
    Hybrid intelligence layer that adjusts ML probability using standard heuristics.
    Prevents false positives on newsletters, digests, and trusted notifications.
    """
    sender_lower = sender.lower()
    subject_lower = subject.lower()
    body_lower = body.lower()
    
    # 1. Trusted Sender keywords
    trusted_keys = ["github", "google", "linkedin", "microsoft", "spotify", "slack", "zoom", "apple", "amazon", "netflix", "openai"]
    is_trusted_sender = any(k in sender_lower for k in trusted_keys)
    
    # 2. Known Safe digest keywords
    digest_signals = [
        "monthly digest", "weekly digest", "activity on github", 
        "repositories you might be interested in", "happy coding!", 
        "unread notifications", "here's your monthly digest", "monthly update"
    ]
    is_digest = any(ds in body_lower or ds in subject_lower for ds in digest_signals)
    
    # 3. Phishing Indicators (Urgency triggers that override whitelist)
    urgency_indicators = [
        "unusual activity", "compromised", "verify your account", 
        "suspend", "action required", "immediate action", "security alert: sign-in"
    ]
    has_urgency = any(trig in body_lower or trig in subject_lower for trig in urgency_indicators)
    
    # Overwrite/adjust rules
    if (is_trusted_sender or is_digest) and not has_urgency:
        # Standard digest/newsletter from trusted platform -> Override to safe!
        adjusted_prob = min(raw_prob, 0.12)  # Limit to 12% probability (Low Risk)
        return adjusted_prob, 0
        
    return raw_prob, (1 if raw_prob >= 0.70 else 0)


def generate_recommendations(is_phishing: bool, signals: dict, prob: float) -> list:
    """Generate security recommendations based on analysis results."""
    recs = []
    if is_phishing:
        recs.append({"icon": "trash-2", "priority": "critical",
                     "title": "Delete Immediately",
                     "body": "Do not click any links, download attachments, or reply to this email."})
        recs.append({"icon": "shield", "priority": "high",
                     "title": "Report as Phishing",
                     "body": "Report this email to your email provider and IT security team."})
        if signals.get("urls", 0) > 2:
            recs.append({"icon": "link-off", "priority": "high",
                         "title": "Suspicious Links Detected",
                         "body": f"Found {signals['urls']} URL patterns. Never click links from untrusted senders."})
        if signals.get("urgency", 0) > 3:
            recs.append({"icon": "clock", "priority": "medium",
                         "title": "Urgency Tactics Detected",
                         "body": "Phishers create urgency to bypass rational thinking. Legitimate organizations rarely demand immediate action."})
        recs.append({"icon": "lock", "priority": "medium",
                     "title": "Secure Your Accounts",
                     "body": "If you clicked any links, change your passwords immediately and enable 2FA."})
    else:
        recs.append({"icon": "check-circle", "priority": "safe",
                     "title": "Email Appears Safe",
                     "body": "Our model did not detect phishing patterns in this email."})
        if prob > 0.25:
            recs.append({"icon": "eye", "priority": "medium",
                         "title": "Exercise Caution",
                         "body": "While classified as safe, some minor indicators were detected. Verify the sender before sharing sensitive information."})
        recs.append({"icon": "shield", "priority": "info",
                     "title": "Stay Vigilant",
                     "body": "Always verify sender identity, avoid unsolicited attachments, and keep security software updated."})
    return recs


def generate_charts(prob: float, signals: dict, is_phishing: bool) -> dict:
    """Generate analysis visualisation charts."""
    charts = {}
    plt.rcParams.update({"font.family": "sans-serif",
                          "axes.spines.top": False, "axes.spines.right": False})

    nav = PALETTE["navy"]
    amb = PALETTE["amber"]
    org = PALETTE["orange"]
    bg  = "#F4F4FF"

    # ── Chart 1: Threat probability gauge ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 1.8), facecolor=bg)
    ax.set_facecolor(bg)
    pct = prob * 100
    color = org if is_phishing else "#22c55e"
    ax.barh([0], [100], color="#E2E2F0", height=0.5)
    ax.barh([0], [pct], color=color,    height=0.5)
    ax.set_xlim(0, 110)
    ax.set_yticks([])
    ax.set_xlabel("Phishing Probability (%)", color=nav, fontsize=9)
    ax.tick_params(colors=nav, labelsize=8)
    ax.text(min(pct + 2, 102), 0, f"{pct:.1f}%", va="center",
            color=nav, fontsize=10, fontweight="bold")
    ax.set_title("Phishing Probability Score", color=nav, fontsize=10, fontweight="bold", pad=8)
    for spine in ax.spines.values(): spine.set_color("#DDDDEE")
    charts["gauge"] = _fig_to_b64(fig)

    # ── Chart 2: Signal indicators bar chart ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor=bg)
    ax.set_facecolor(bg)
    labels  = ["URL Count", "Urgency Words", "Suspicious Phrases", "Email Addresses", "Special Chars"]
    values  = [
        min(signals.get("urls", 0) / 10, 1),
        min(signals.get("urgency", 0) / 10, 1),
        min(signals.get("suspicious", 0) / 8, 1),
        min(signals.get("email_addrs", 0) / 5, 1),
        min(signals.get("special_chars", 0) / 20, 1),
    ]
    bar_cols = [org if v > 0.4 else amb if v > 0.15 else "#22c55e" for v in values]
    bars = ax.barh(labels, values, color=bar_cols, height=0.55)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Normalised Signal Strength", color=nav, fontsize=8)
    ax.set_title("Threat Signal Analysis", color=nav, fontsize=10, fontweight="bold")
    ax.tick_params(colors=nav, labelsize=8)
    for bar, v in zip(bars, values):
        ax.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", color=nav, fontsize=8)
    for spine in ax.spines.values(): spine.set_color("#DDDDEE")
    charts["signals"] = _fig_to_b64(fig)

    # ── Chart 3: Safe vs Phishing confidence pie ───────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor=bg)
    ax.set_facecolor(bg)
    sizes   = [1 - prob, prob]
    clrs    = ["#22c55e", org]
    lbls    = ["Safe", "Phishing"]
    explode = (0, 0.05) if is_phishing else (0.05, 0)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=lbls, colors=clrs, autopct="%1.1f%%",
        explode=explode, startangle=90,
        textprops={"color": nav, "fontsize": 9},
        wedgeprops={"edgecolor": bg, "linewidth": 2},
    )
    for at in autotexts: at.set_fontweight("bold")
    ax.set_title("Classification Confidence", color=nav, fontsize=10, fontweight="bold", pad=12)
    charts["pie"] = _fig_to_b64(fig)

    return charts


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def intro():
    return render_template("intro.html")


@app.route("/onboard/<int:step>")
def onboard(step):
    if step < 1 or step > 3:
        return redirect(url_for("intro"))
    return render_template(f"onboard{step}.html")


@app.route("/dashboard")
def dashboard():
    stats = {
        "model_accuracy": metadata.get("accuracy", 98.26),
        "roc_auc": metadata.get("roc_auc", 0.9969),
        "train_samples": metadata.get("train_samples", 14476),
        "total_samples": metadata.get("total_samples", 18096),
        "vocab_size": metadata.get("vocab_size", 40000),
    }
    return render_template("dashboard.html", stats=stats)


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "GET":
        return render_template("analyze.html")

    # POST – analyse submitted text
    email_text = request.form.get("email_text", "").strip()
    email_subject = request.form.get("email_subject", "").strip()
    sender_name   = request.form.get("sender_name", "Unknown Sender").strip()

    errors = []
    if not email_text or len(email_text) < 20:
        errors.append("Please enter a valid email body (at least 20 characters).")
    if errors:
        return render_template("analyze.html", errors=errors,
                               email_text=email_text,
                               email_subject=email_subject,
                               sender_name=sender_name)

    # Preprocess & predict
    clean   = preprocess_text(email_text)
    raw_prob = float(pipeline.predict_proba([clean])[0][1])  # raw ML probability
    
    # Refine prediction with Hybrid Rules Engine (avoids newsletters/trusted sender false positives)
    prob, pred = adjust_threat_with_rules(sender_name, email_subject, email_text, raw_prob)
    is_phishing = bool(pred == 1)
    label   = "Phishing Email" if is_phishing else "Safe Email"

    # Signals, threat level, recommendations
    signals       = get_risk_signals(email_text)
    threat        = compute_threat_level(prob)
    recommendations = generate_recommendations(is_phishing, signals, prob)
    charts        = generate_charts(prob, signals, is_phishing)

    # Store in session for back-navigation
    session["last_result"] = {
        "label": label, "prob": round(prob * 100, 1),
        "is_phishing": is_phishing,
        "subject": email_subject or "(No subject)",
        "sender": sender_name,
        "timestamp": datetime.now().strftime("%d %B %Y, %H:%M"),
    }

    return render_template(
        "results.html",
        label        = label,
        is_phishing  = is_phishing,
        probability  = round(prob * 100, 1),
        threat       = threat,
        signals      = signals,
        recommendations = recommendations,
        charts       = charts,
        email_text   = email_text[:500] + ("…" if len(email_text) > 500 else ""),
        email_subject = email_subject or "(No subject)",
        sender_name  = sender_name,
        now          = datetime.now().strftime("%d %B %Y, %H:%M"),
        metadata     = metadata,
    )


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """JSON API endpoint for Gmail integration demo."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    clean = preprocess_text(text)
    prob  = float(pipeline.predict_proba([clean])[0][1])
    pred  = int(pipeline.predict([clean])[0])
    return jsonify({
        "label": "Phishing Email" if pred == 1 else "Safe Email",
        "is_phishing": pred == 1,
        "probability": round(prob * 100, 1),
        "threat_level": compute_threat_level(prob)["level"],
    })


# ── Gmail API Settings ─────────────────────────────────────────────────────────
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # ONLY for local development!
CLIENT_SECRETS_FILE = os.path.join(BASE_DIR, "credentials.json")
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

@app.route("/gmail")
def gmail_integration():
    if "gmail_creds" in session:
        try:
            creds = Credentials(**session["gmail_creds"])
            # Refresh token if expired
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                session["gmail_creds"] = {
                    "token": creds.token,
                    "refresh_token": creds.refresh_token,
                    "token_uri": creds.token_uri,
                    "client_id": creds.client_id,
                    "client_secret": creds.client_secret,
                    "scopes": creds.scopes
                }

            # Get credentials dict from session to safely pass to concurrent threads
            creds_dict = session.get("gmail_creds")

            # Connect to Gmail Service (main thread)
            service = build("gmail", "v1", credentials=creds)
            
            # Fetch 50 latest primary inbox messages
            results = service.users().messages().list(userId="me", maxResults=50, q="category:primary").execute()
            messages = results.get("messages", [])
            
            from concurrent.futures import ThreadPoolExecutor
            
            def fetch_and_scan_email(m):
                try:
                    # Create a completely isolated copy of Credentials and service for this thread
                    thread_creds = Credentials(**creds_dict)
                    thread_service = build("gmail", "v1", credentials=thread_creds, static_discovery=True)
                    
                    # Fetch detailed message object (format="full") - 100% accurate context!
                    msg = thread_service.users().messages().get(
                        userId="me", 
                        id=m["id"], 
                        format="full"
                    ).execute()
                    
                    payload = msg.get("payload", {})
                    headers = payload.get("headers", [])
                    
                    # Parse Sender and Subject
                    subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "(No Subject)")
                    sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "Unknown Sender")
                    
                    # Extract plain text body
                    body = ""
                    if "parts" in payload:
                        for part in payload["parts"]:
                            if part["mimeType"] == "text/plain" and "data" in part["body"]:
                                body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                                break
                            elif "parts" in part:
                                for subpart in part["parts"]:
                                    if subpart["mimeType"] == "text/plain" and "data" in subpart["body"]:
                                        body = base64.urlsafe_b64decode(subpart["body"]["data"]).decode("utf-8")
                                        break
                    elif "body" in payload and "data" in payload["body"]:
                        body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")

                    if not body:
                        body = msg.get("snippet", "")

                    # Predict Phishing risk
                    clean_body = preprocess_text(body)
                    raw_prob = float(pipeline.predict_proba([clean_body])[0][1])
                    
                    # Refine threat level with hybrid rule assessor
                    prob, pred = adjust_threat_with_rules(sender, subject, body, raw_prob)
                    
                    threat_info = compute_threat_level(prob)
                    
                    return {
                        "id": m["id"],
                        "sender": sender,
                        "subject": subject,
                        "probability": round(prob * 100, 1),
                        "is_phishing": pred == 1,
                        "threat_level": threat_info["level"],
                        "color": threat_info["color"],
                        "bg": threat_info["bg"],
                        "border": threat_info["border"],
                        "icon": threat_info["icon"],
                        "snippet": body[:120] + "..." if len(body) > 120 else body
                    }
                except Exception as ex:
                    print(f"Error parsing message {m['id']}: {ex}")
                    return None

            # Execute the Gmail fetching & analysis concurrently using 20 threads!
            scanned_emails = []
            with ThreadPoolExecutor(max_workers=20) as executor:
                results_list = list(executor.map(fetch_and_scan_email, messages))
                
            # Filter out any failed requests
            scanned_emails = [r for r in results_list if r is not None]

            return render_template("gmail.html", connected=True, emails=scanned_emails)
            
        except Exception as e:
            print("Gmail API Error:", e)
            session.pop("gmail_creds", None)

    return render_template("gmail.html", connected=False)

@app.route("/gmail/login")
def gmail_login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri="http://127.0.0.1:5001/gmail/callback"
    )
    authorization_url, state = flow.authorization_url(access_type="offline", include_granted_scopes="true")
    session["state"] = state
    session["code_verifier"] = flow.code_verifier  # Store the auto-generated code verifier in session!
    return redirect(authorization_url)

@app.route("/gmail/callback")
def gmail_callback():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        state=session["state"],
        redirect_uri="http://127.0.0.1:5001/gmail/callback"
    )
    flow.code_verifier = session.get("code_verifier")  # Restore the code verifier before fetching the token!
    flow.fetch_token(authorization_response=request.url)
    
    creds = flow.credentials
    session["gmail_creds"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes
    }
    return redirect(url_for("gmail_integration"))

@app.route("/gmail/logout")
def gmail_logout():
    session.pop("gmail_creds", None)
    return redirect(url_for("gmail_integration"))



@app.route("/settings")
def settings():
    return render_template("settings.html")


@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("intro"))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
