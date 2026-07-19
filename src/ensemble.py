import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


def wape(actual, predicted):
    return (abs(actual - predicted).sum() / actual.sum()) * 100


def build_ensemble(xgb_pred_path='../outputs/xgboost_predictions.csv',
                    prophet_pred_path='../outputs/prophet_predictions.csv',
                    category_name='Camping & Hiking'):

    xgb_df = pd.read_csv(xgb_pred_path, parse_dates=['order_date'])
    prophet_df = pd.read_csv(prophet_pred_path, parse_dates=['ds'])
    prophet_df = prophet_df.rename(columns={'ds': 'order_date', 'yhat': 'prophet_pred', 'y': 'actual'})

    # Only keep XGBoost rows for the SAME category Prophet was trained on
    xgb_cat = xgb_df[xgb_df['Category Name'] == category_name]

    if len(xgb_cat) == 0:
        print(f"'{category_name}' not found in XGBoost test set.")
        print("Available categories in XGBoost test set:")
        print(xgb_df['Category Name'].unique())
        return None, None

    merged = xgb_cat.merge(
        prophet_df[['order_date', 'prophet_pred']],
        on='order_date', how='inner'
    )

    if len(merged) == 0:
        print("No overlapping dates between XGBoost test set and Prophet predictions for this category.")
        return None, None

    print(f"Comparing on {len(merged)} overlapping rows (category: {category_name})\n")

    best_wape = float('inf')
    best_weight = None

    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        ensemble_pred = w * merged['xgb_pred'] + (1 - w) * merged['prophet_pred']
        mape = mean_absolute_percentage_error(merged['quantity'], ensemble_pred) * 100
        wape_score = wape(merged['quantity'], ensemble_pred)
        print(f"XGBoost weight {w:.1f} -> MAPE: {mape:.2f}%  WAPE: {wape_score:.2f}%")
        if wape_score < best_wape:
            best_wape = wape_score
            best_weight = w

    print(f"\nBest weight: {best_weight}, Best Ensemble WAPE: {best_wape:.2f}%")

    # For comparison — how did each model do alone on this exact same set of rows?
    xgb_only_wape = wape(merged['quantity'], merged['xgb_pred'])
    prophet_only_wape = wape(merged['quantity'], merged['prophet_pred'])
    print(f"\nXGBoost alone WAPE:  {xgb_only_wape:.2f}%")
    print(f"Prophet alone WAPE:  {prophet_only_wape:.2f}%")
    print(f"Best ensemble WAPE:  {best_wape:.2f}%")

    return best_weight, best_wape


if __name__ == "__main__":
    build_ensemble()