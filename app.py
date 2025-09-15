import gradio as gr
import pandas as pd
import numpy as np
import json
from pathlib import Path
import lightgbm as lgb

ART_DIR = Path("artifacts")
MODEL_TXT = ART_DIR / "creditcard_lgb.txt"
FEATURES_JSON = ART_DIR / "feature_columns.json"
SAMPLE_CSV = ART_DIR / "sample_transactions.csv"

# ---- Load artifacts ----
with open(FEATURES_JSON) as f:
    FEATURE_COLS = json.load(f)

gbm = lgb.Booster(model_file=str(MODEL_TXT))

# Sample table for the UI
sample_df = pd.read_csv(SAMPLE_CSV)
# Optional: limit to 100 to keep the UI snappy
sample_df = sample_df.head(100)

def preprocess(amount, time, pca_features_str):
    """
    Build a single-row DataFrame with all features expected by the model.
    We expect pca_features_str as a comma-separated string for V1..V28.
    """
    # Parse PCA features
    vals = [v.strip() for v in pca_features_str.split(",") if v.strip() != ""]
    try:
        vals = [float(x) for x in vals]
    except Exception:
        raise gr.Error("PCA features must be numeric, comma-separated values")

    if len(vals) != 28:
        raise gr.Error(f"Expected 28 PCA features (V1..V28). Got {len(vals)}")

    # Map to V1..V28
    vcols = [f"V{i}" for i in range(1, 29)]
    row = dict(zip(vcols, vals))

    # Time / Amount
    row["Time"] = float(time)
    row["Amount"] = float(amount)

    # Ensure the model columns are present
    df = pd.DataFrame([row], columns=FEATURE_COLS).fillna(0.0)
    return df

def predict_fraud(amount, time, pca_features_str, threshold=0.5):
    X = preprocess(amount, time, pca_features_str)
    prob = float(gbm.predict(X)[0])
    decision = "FRAUD" if prob >= threshold else "LEGIT"
    return f"Probability: {prob:.4f}  |  Decision: {decision} (thr={threshold:.2f})"

def use_example(example_idx):
    """
    Returns Amount, Time, PCA features string for the chosen example row.
    """
    idx = int(example_idx)
    if idx < 0 or idx >= len(sample_df):
        raise gr.Error("Example index out of range")

    row = sample_df.iloc[idx]
    amount = row.get("Amount", 0.0)
    time = row.get("Time", 0.0)

    # Build PCA features string (V1..V28)
    vcols = [f"V{i}" for i in range(1, 29)]
    feats = [str(row[c]) for c in vcols]
    feat_str = ", ".join(feats)
    return str(amount), str(time), feat_str

with gr.Blocks(theme="gradio/soft") as demo:
    gr.Markdown("## 💳 Credit Card Fraud Detection (LightGBM)")
    gr.Markdown("Enter a transaction or pick an example. The model returns the fraud probability and decision.")

    with gr.Row():
        amount_in = gr.Textbox(label="Transaction Amount", placeholder="e.g., 123.45")
        time_in = gr.Textbox(label="Time (seconds since first tx)", value="0")

    pca_in = gr.Textbox(
        label="PCA Features (comma separated V1..V28)",
        placeholder="e.g., -1.23, 0.45, 0.67, ... exactly 28 values"
    )

    threshold = gr.Slider(0.0, 1.0, value=0.50, step=0.01, label="Decision threshold")

    predict_btn = gr.Button("Predict", variant="primary")
    pred_out = gr.Textbox(label="Prediction", interactive=False)

    gr.Markdown("### Example Transactions")
    with gr.Row():
        example_idx = gr.Dropdown(
            choices=[str(i) for i in sample_df.index],
            label="Pick example #",
            value=str(sample_df.index[0]) if len(sample_df) else None
        )
        use_btn = gr.Button("Use example")

    # Show the sample table for transparency
    table = gr.Dataframe(
        value=sample_df[["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]],
        wrap=True,
        label="Example Transactions (first 100)"
    )

    # Wire events
    use_btn.click(
        fn=use_example,
        inputs=[example_idx],
        outputs=[amount_in, time_in, pca_in]
    )

    predict_btn.click(
        fn=predict_fraud,
        inputs=[amount_in, time_in, pca_in, threshold],
        outputs=[pred_out]
    )

demo.launch()
