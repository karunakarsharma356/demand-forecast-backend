import pandas as pd
import numpy as np


def build_features(df, category_col='Category Name', date_col='order_date', target_col='quantity'):
    """
    Takes daily aggregated demand data and builds lag/rolling/calendar features.
    Expects one row per (category, date).
    """
    df = df.copy()
    df = df.sort_values([category_col, date_col])

    # Calendar features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Lag features (per category, so no leakage across categories)
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = df.groupby(category_col)[target_col].shift(lag)

    # Rolling averages (shifted first to avoid leakage)
    # Rolling averages (shifted first to avoid leakage)
    for window in [7, 30]:
        df[f'rolling_mean_{window}'] = (
            df.groupby(category_col)[target_col]
              .shift(1)
              .rolling(window)
              .mean()
        )

    return df


def encode_category(df, category_col='Category Name'):
    """Converts category names into numeric codes for XGBoost."""
    df = df.copy()
    df[f'{category_col}_code'] = df[category_col].astype('category').cat.codes
    return df