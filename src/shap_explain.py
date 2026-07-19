import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(__file__))
from features import build_features, encode_category

FEATURE_COLS = [
    'Category Name_code', 'day_of_week', 'is_weekend', 'month', 'week_of_year',
    'lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_mean_30',
    'avg_discount_rate', 'late_delivery_risk'
]


def explain():
    model = joblib.load('../models/xgboost_model.pkl')

    df = pd.read_csv('../data/daily_demand_by_category.csv', parse_dates=['order_date'])

    # Match the same filtering used in training, so SHAP sees realistic data
    category_totals = df.groupby('Category Name')['quantity'].sum().sort_values(ascending=False)
    top_categories = category_totals.head(15).index.tolist()
    df = df[df['Category Name'].isin(top_categories)]
    df = df[df['quantity'] >= 15]

    df = build_features(df)
    df = encode_category(df)
    df = df.dropna(subset=FEATURE_COLS)

    X_sample = df[FEATURE_COLS].sample(n=min(2000, len(df)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_COLS, show=False)
    plt.tight_layout()
    plt.savefig('../outputs/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved SHAP summary plot to outputs/shap_summary.png")

    # Also print mean absolute SHAP value per feature — a simple ranked importance table
    import numpy as np
    mean_abs_shap = pd.DataFrame({
        'feature': FEATURE_COLS,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    print("\nFeature importance (by mean |SHAP value|):")
    print(mean_abs_shap.to_string(index=False))


if __name__ == "__main__":
    explain()