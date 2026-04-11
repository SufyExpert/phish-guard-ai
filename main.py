"""
main.py — Entry point for the Health Risk Prediction System.

Run:
  Step 1 (first time only): python train_model.py
  Step 2:                   python main.py
  Then open:                http://localhost:5000
"""

from app import app

if __name__ == "__main__":
    import os, sys

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    required  = ["rf_model.pkl", "scaler.pkl", "feature_columns.pkl",
                 "le_risk.pkl",  "mappings.pkl"]

    missing = [f for f in required
               if not os.path.exists(os.path.join(model_dir, f))]

    if missing:
        print("="*58)
        print("  MODEL ARTEFACTS NOT FOUND — running train_model.py first")
        print("="*58)
        import subprocess
        ret = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "train_model.py")],
            check=True
        )

    print("\n" + "="*58)
    print("  Health Risk Prediction System")
    print("  Visit: http://localhost:5000")
    print("="*58 + "\n")

    app.run(debug=False, port=5000, host="0.0.0.0")
