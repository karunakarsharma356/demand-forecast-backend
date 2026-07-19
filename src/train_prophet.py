import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import joblib


def wape(actual, predicted):
    return (abs(actual - predicted).sum() / actual.sum()) * 100


def train_prophet_for_category(daily_df, category_name):
    cat_df = daily_df[daily_df['Category Name'] == category_name][['order_date', 'quantity']]
    cat_df = cat_df.rename(columns={'order_date': 'ds', 'quantity': 'y'})
    cat_df = cat_df.sort_values('ds')

    cutoff = cat_df['ds'].quantile(0.8)
    train_df = cat_df[cat_df['ds'] <= cutoff]
    test_df = cat_df[cat_df['ds'] > cutoff]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(train_df)

    future = test_df[['ds']]
    forecast = model.predict(future)

    merged = test_df.merge(forecast[['ds', 'yhat']], on='ds')
    merged['yhat'] = merged['yhat'].clip(lower=0)

    mape = mean_absolute_percentage_error(merged['y'], merged['yhat']) * 100
    wape_score = wape(merged['y'], merged['yhat'])

    print(f"Prophet MAPE for '{category_name}': {mape:.2f}%")
    print(f"Prophet WAPE for '{category_name}': {wape_score:.2f}%")

    return model, merged


if __name__ == "__main__":
    daily = pd.read_csv('../data/daily_demand_by_category.csv', parse_dates=['order_date'])

    top_category = daily['Category Name'].value_counts().index[0]
    print(f"Training Prophet on top category: {top_category}")

    model, results = train_prophet_for_category(daily, top_category)

    joblib.dump(model, '../models/prophet_model.pkl')
    results.to_csv('../outputs/prophet_predictions.csv', index=False)