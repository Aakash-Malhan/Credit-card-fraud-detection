# Credit-card-fraud-detection
This project implements a credit card fraud detection system using the popular Credit Card Fraud Dataset (Europe, 2013 – 284,807 transactions). The model is trained with LightGBM to handle class imbalance and achieve high precision/recall on fraudulent transactions. # Credit Card Fraud Detection (2013 dataset, LightGBM)

Live demo: https://huggingface.co/spaces/aakash-malhan/aakash-malhan_creditcard_fraud_lgbm

- Trained on the classic Credit Card Fraud (2013) dataset (284,807 tx, highly imbalanced).
- Model: LightGBM on PCA features (V1..V28), with threshold tuning for fraud recall.
- Includes a Gradio app to score transactions and try examples.

## Metrics (test split)
| Model      | PR-AUC | ROC-AUC | Precision@K | Recall@K |
|------------|:------:|:-------:|:-----------:|:--------:|
| LightGBM   | 0.8737 | 0.9626  |    0.7522   |  0.8586  |
| Logistic   | 0.7562 | 0.9649  |    0.7257   |  0.8288  |
| ISOForest  | 0.0323 | 0.4617  |     NaN     |   0.0879 |
| LOF        | 0.3484 | 0.5331  |     NaN     |   0.3984 |

**Artifacts** (under `/artifacts`):
- `creditcard_lgb.txt` — trained model
- `feature_columns.json` — exact feature order
- `sample_transactions.csv` — quick test examples
- `creditcard_metrics.json` — run metrics
- `creditcard_feature_importance.csv` — feature importance (plot in Colab)

## Try it
1. Open the live demo.
2. Either paste your own values or click **Use example** and **Predict**.
3. The app shows **Probability** and **Decision** based on the threshold slider.

> Notes: Dataset is anonymized PCA components; the app accepts **V1..V28**, **Time**, **Amount**.




