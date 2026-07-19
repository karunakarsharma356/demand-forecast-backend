from fastapi import FastAPI, HTTPException
import pandas as pd
import xgboost as xgb
import joblib
import sys
import os

sys.path.append(os.path.dirname(__file__))
from features import build_features, encode_category
from inventory_optimization import build_inventory_plan

app = FastAPI(title="AI Demand Forecasting & Inventory Optimization API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = xgb.XGBRegressor()
model.load_model('../models/xgboost_model.json')

daily_demand = pd.read_csv('../data/daily_demand_by_category.csv', parse_dates=['order_date'])

FEATURE_COLS = [
    'Category Name_code', 'day_of_week', 'is_weekend', 'month', 'week_of_year',
    'lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_mean_30',
    'avg_discount_rate', 'late_delivery_risk'
]


@app.get("/")
def root():
    return {"message": "Demand Forecasting & Inventory Optimization API is running"}


@app.get("/categories")
def get_categories():
    categories = daily_demand['Category Name'].unique().tolist()
    return {"categories": categories}


@app.get("/forecast/{category_name}")
def get_forecast(category_name: str, days: int = 7):
    cat_df = daily_demand[daily_demand['Category Name'] == category_name].copy()
    raw_row_count = len(cat_df)

    if raw_row_count == 0:
        raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found")

    cat_df = build_features(cat_df)
    cat_df = encode_category(cat_df, category_col='Category Name')

    cat_df[['avg_discount_rate', 'late_delivery_risk']] = (
        cat_df[['avg_discount_rate', 'late_delivery_risk']].fillna(0)
    )

    cat_df = cat_df.dropna(subset=FEATURE_COLS)

    if len(cat_df) == 0:
        raise HTTPException(status_code=400, detail="Not enough history to build features for this category")

    latest_row = cat_df.iloc[[-1]][FEATURE_COLS]
    prediction = model.predict(latest_row)[0]

    return {
        "category": category_name,
        "predicted_next_day_demand": round(float(prediction), 1),
        "last_actual_demand": float(cat_df.iloc[-1]['quantity']),
        "last_date": str(cat_df.iloc[-1]['order_date'].date())
    }


@app.get("/inventory/{category_name}")
def get_inventory_plan(category_name: str, lead_time_days: int = 7, service_level: float = 0.95):
    plan = build_inventory_plan(
        daily_demand, category_name,
        lead_time_days=lead_time_days,
        service_level=service_level
    )

    if "error" in plan:
        raise HTTPException(status_code=404, detail=plan["error"])

    return plan


@app.get("/dashboard-summary")
def dashboard_summary():
    top_categories = (
        daily_demand.groupby('Category Name')['quantity']
        .sum().sort_values(ascending=False).head(15).index.tolist()
    )

    results = []
    for cat in top_categories:
        try:
            plan = build_inventory_plan(daily_demand, cat, lead_time_days=7, service_level=0.95)
            results.append(plan)
        except Exception:
            continue

    return {"summary": results}