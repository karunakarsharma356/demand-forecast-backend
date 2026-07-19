import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import joblib
import sys
import os

sys.path.append(os.path.dirname(__file__))
from features import build_features, encode_category


FEATURE_COLS = [
    'Category Name_code', 'day_of_week', 'is_weekend', 'month', 'week_of_year',
    'lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_mean_30',
    'avg_discount_rate', 'late_delivery_risk'
]
TARGET_COL = 'quantity'


def load_and_prepare():
    df = pd.read_csv('../data/daily_demand_by_category.csv', parse_dates=['order_date'])

    category_totals = df.groupby('Category Name')['quantity'].sum().sort_values(ascending=False)
    top_categories = category_totals.head(15).index.tolist()
    df = df[df['Category Name'].isin(top_categories)]

    df = df[df['quantity'] >= 15]

    df = build_features(df)
    df = encode_category(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df


def wape(actual, predicted):
    return (abs(actual - predicted).sum() / actual.sum()) * 100


def train():
    df = load_and_prepare()

    cutoff = df['order_date'].quantile(0.8)
    train_df = df[df['order_date'] <= cutoff]
    test_df = df[df['order_date'] > cutoff]

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=30
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    preds = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, preds) * 100
    mae = mean_absolute_error(y_test, preds)
    wape_score = wape(y_test, preds)

    print(f"\nXGBoost Test MAPE: {mape:.2f}%")
    print(f"XGBoost Test MAE:  {mae:.2f} units")
    print(f"XGBoost Test WAPE: {wape_score:.2f}%")

    diag = test_df.copy()
    diag['pred'] = preds
    diag['abs_pct_error'] = (abs(diag['quantity'] - diag['pred']) / diag['quantity']) * 100

    print("\nWorst 10 predictions (by % error):")
    print(diag[['Category Name', 'order_date', 'quantity', 'pred', 'abs_pct_error']]
          .sort_values('abs_pct_error', ascending=False).head(10))

    print("\n--- Cardio Equipment monthly avg quantity (full history) ---")
    cardio = df[df['Category Name'] == 'Cardio Equipment'].copy()
    cardio['month_year'] = cardio['order_date'].dt.to_period('M')
    print(cardio.groupby('month_year')['quantity'].mean())

    # Save in both formats — pickle for quick local reuse, JSON for stable cross-version loading
    joblib.dump(model, '../models/xgboost_model.pkl')
    model.save_model('../models/xgboost_model.json')

    test_df = test_df.copy()
    test_df['xgb_pred'] = preds
    test_df.to_csv('../outputs/xgboost_predictions.csv', index=False)

    return model, test_df


if __name__ == "__main__":
    train()