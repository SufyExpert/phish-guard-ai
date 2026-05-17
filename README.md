# 🛡️ PhishGuard AI — Advanced Email Phishing Detection System

PhishGuard AI is a state-of-the-art, machine-learning-powered web application that identifies email phishing attempts with **98.26% accuracy** and **0.9969 ROC-AUC**. Integrating advanced NLP (TF-IDF Vectorization) with a tuned Logistic Regression model, PhishGuard AI features a hybrid intelligence system: combining AI model predictions with a secure heuristics rules engine to completely eliminate false positives on newsletters, notifications, and monthly digests from trusted platforms (such as GitHub, Google, Slack, and Microsoft).

The application features a complete **Gmail Active Scanner** that queries your Gmail inbox concurrently via multi-threaded OAuth2 services, analyzes threats in real-time, displays interactive threat details in glassmorphic popups, and deep-links directly to individual emails in the live Google Gmail interface.

---

## 📸 Core Features & Premium UX

### 🔍 1. Real-Time Text Analyzer
* Simply paste raw email content to evaluate risk percentages instantly.
* Displays a detailed threat signal analysis report showing URL densities, urgency indicators, suspicious phrases, special character counts, and statistical classification charts.
* Autogenerates actionable, priority-sorted security recommendations (e.g. Delete, Verify domain, Check attachments).

### ⚡ 2. High-Performance Multi-Threaded Gmail Active Scanner
* Connects seamlessly via Google OAuth2 with secure read-only access.
* **Concurrent Scanning Pipeline:** Instantiates isolated credentials and Gmail service clients concurrently using a 20-worker `ThreadPoolExecutor` to fetch and scan 50 emails in **under 2 seconds**!
* Displays a live scan interface showing recent primary inbox mail rows styled dynamically according to threat risk (Critical, High, Medium, Safe).

### 🖥️ 3. Premium Glassmorphic Threat Details Modal & Deep Links
* Clicking on any threat row in the scanner opens an elegant details popup modal featuring dynamic threat color matching, analyzed content formatting, and security details.
* Features a direct inline shortcut button **"Open in Gmail"** that deep-links directly into your live Gmail web interface targeting the specific email message ID in a new browser tab (`https://mail.google.com/mail/u/0/#all/{message_id}`).

### 🛡️ 4. Hybrid Security Rules Engine
* Incorporates a dual-layer cybersecurity heuristics engine alongside the ML model.
* Automatically whitelists and safe-checks legitimate notifications, monthly updates, and digests from verified senders (e.g. `@github.com`, `@google.com`, `@linkedin.com`) to **completely prevent false positives**.
* Integrates a calibrated decision threshold of **70% confidence** for classification, mimicking professional, industry-standard enterprise firewalls.

---

## 🛠️ Tech Stack & Model Metrics

| Component     | Technology / Library                                 | Details / Performance Metrics                     |
|---------------|-----------------------------------------------------|---------------------------------------------------|
| **Backend**   | Python, Flask                                       | Dynamic routing, stateful session, secure OAuth   |
| **Frontend**  | HTML5 Semantic Layout, CSS3 (Glassmorphism & Gradients), Jinja2 | Modern Outfit & Inter typography, Feather Icons  |
| **ML Pipeline**| scikit-learn (Logistic Regression, TF-IDF)          | Log-scaled TF-IDF (40,000 features, N-gram (1,2))|
| **Accuracy**  | **98.26%**                                          | High-fidelity classification on 18,096 samples   |
| **ROC-AUC**   | **0.9969**                                          | Stellar class separation performance             |
| **Analytics** | Matplotlib                                          | Dynamic threat gauges, signal horizontal bars, pie charts served inline as base64 PNGs |
| **Concurrency**| `concurrent.futures.ThreadPoolExecutor` (20 Workers) | Thread-safe, isolated API clients                |
| **Auth & API**| `google-auth-oauthlib`, `googleapiclient`           | Google OAuth2 & Gmail API integrations            |

---

## 🧠 Machine Learning & Hybrid Rules Heuristic

### A. The ML Inference Pipeline
1. **Text Cleansing:** Text is converted to lowercase; HTML tags, non-alphabetic characters, and noisy symbols are stripped. Specific URLs and email addresses are replaced with standard token tags (`url`, `email`).
2. **Vectorization:** The processed string is projected into a 40,000-dimensional space using a TF-IDF vectorizer trained on unigrams and bigrams.
3. **Classification:** A balanced Logistic Regression classifier computes the probability boundary.

### B. Heuristic Layer (Domain & Urgent Scanner)
No pure ML model is perfectly suited for newsletters or weekly updates without false positives. PhishGuard AI solves this with a **Reputation Heuristic Layer**:
```python
# Senders like github.com/google.com are checked for standard notifications.
# If no critical panic triggers (e.g. "compromised", "verify your account", "suspend") are found,
# standard notifications are capped at 12% probability (Low Risk) to prevent ML false alarms.
```

---

## 📁 Native Directory Structure

```
phishguard-ai/
│
├── app.py                      # Core Flask Application & Threat Analysis Logic
├── train_model.py              # ML Training Pipeline & Model Serialization
│
├── model/                      # Serialized ML Model Artifacts (Joblib dumps)
│   ├── phishing_pipeline.pkl   # Fitted TF-IDF Vectorizer + Logistic Regression pipeline
│   └── metadata.pkl            # Accuracy, ROC-AUC, and feature metadata metrics
│
├── templates/                  # Flask HTML5 templates using Jinja2
│   ├── base.html               # Main dashboard framework & gradient SVG favicon
│   ├── intro.html              # Gorgeous modern onboarding intro portal
│   ├── onboard1.html           # Onboarding Step 1 (Model pipelines explanation)
│   ├── onboard2.html           # Onboarding Step 2 (Threat classification description)
│   ├── onboard3.html           # Onboarding Step 3 (Launch checklist)
│   ├── dashboard.html          # Stats dashboard & model details
│   ├── analyze.html            # Copypaste real-time text analyzer form
│   ├── results.html            # Visual analysis reports & dynamic charts
│   ├── gmail.html              # Dynamic multi-threaded Gmail scanner & popup modals
│   └── settings.html           # Simulation settings & metadata controls
│
├── static/                     # Static UI styling and assets
│   └── css/
│       └── style.css           # Premium dark navy & gradient orange stylesheet
│
├── credentials.json            # Google OAuth Web client credentials (User-supplied)
└── Phishing_Email.csv          # Training Dataset (52 MB, ~18.6K items)
```

---

## ⚙️ Quick Start & Setup Guide

### 1. Prerequisites & Dependencies
Ensure Python 3.8+ is installed on your system. Open terminal/PowerShell and install the required dependencies:
```bash
pip install flask scikit-learn pandas numpy matplotlib joblib google-auth-oauthlib google-api-python-client
```

### 2. Configure Google Cloud Platform Console (OAuth Setup)
To connect your Gmail to the active scanner, you must register a Web Application in the Google Cloud Console:
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project called `PhishGuard AI`.
3. Navigate to **APIs & Services** > **Library** and enable the **Gmail API**.
4. Go to **OAuth Consent Screen**:
   * Set User Type to **External**.
   * Add required scopes: `.../auth/gmail.readonly`.
   * **CRITICAL:** Add your testing Gmail address in the **Test Users** section (this bypasses Google's App Verification checks).
5. Navigate to **Credentials** > **Create Credentials** > **OAuth Client ID**:
   * Select application type **Web Application**.
   * Add **Authorized Redirect URIs**:
     ```
     http://127.0.0.1:5001/gmail/callback
     ```
6. Download the generated client JSON credentials, rename the file exactly to `credentials.json`, and place it in the root workspace directory.

### 3. Model Training (Optional/First Time)
To train or re-fit the Logistic Regression classifier on the `Phishing_Email.csv` dataset and serialize model pipelines into `/model/`:
```bash
python train_model.py
```

### 4. Run the Application
Start the Flask local development server:
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5001
```

---

## 🎨 Visual Identity & Brand System

PhishGuard AI features a curated, high-contrast, premium color system designed to excite users and emphasize visual security:

```python
PALETTE = {
    "white":      "#FFFFFF",              # Sleek base layout background
    "amber":      "#FBAC41",              # Warnings and medium-threat highlights
    "orange":     "#FF6632",              # PhishGuard signature brand & accent
    "navy":       "#070033",              # Rich deep primary typography & headers
    "navy_mid":   "#0D0050",              # Background panels & navigation
    "accent_bg":  "rgba(255, 102, 50, 0.12)" # Dynamic translucent warning blocks
}
```

* **Custom Vector Favicon:** Embedded dynamic SVG gradient orange rounded card featuring a crisp white outline shield, perfectly representing the PhishGuard identity.
* **Responsive Layouts:** The workspace is built with dynamic grid models (`gmail-grid`) that fluidly adjust based on connection state and device viewports.
