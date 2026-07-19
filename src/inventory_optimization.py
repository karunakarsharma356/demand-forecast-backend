import pandas as pd
import numpy as np
from scipy.stats import norm


# Service level Z-scores (how confident you want to be against stockouts)
# 95% service level = 1.65, 99% = 2.33, etc.
SERVICE_LEVEL_Z = {
    0.90: 1.28,
    0.95: 1.65,
    0.97: 1.88,
    0.99: 2.33
}


def calculate_safety_stock(demand_std, lead_time_days, service_level=0.95):
    """
    Safety stock formula (Silver, Pyke & Thomas, 2017):
    SS = Z * demand_std * sqrt(lead_time)

    demand_std: standard deviation of daily demand
    lead_time_days: how many days it takes for a reorder to arrive
    service_level: probability of NOT stocking out (0.95 = 95% confidence)
    """
    z = SERVICE_LEVEL_Z.get(service_level, 1.65)
    safety_stock = z * demand_std * np.sqrt(lead_time_days)
    return round(safety_stock, 1)


def calculate_reorder_point(avg_daily_demand, lead_time_days, safety_stock):
    """
    Reorder Point (ROP) = (average daily demand * lead time) + safety stock
    This is the inventory level at which you must place a new order.
    """
    rop = (avg_daily_demand * lead_time_days) + safety_stock
    return round(rop, 1)


def calculate_eoq(annual_demand, order_cost, holding_cost_per_unit):
    """
    Economic Order Quantity (Harris, 1913):
    EOQ = sqrt( (2 * annual_demand * order_cost) / holding_cost_per_unit )

    order_cost: fixed cost per order placed (e.g. $50 admin/shipping cost)
    holding_cost_per_unit: annual cost to hold one unit in inventory (e.g. $2/unit/year)
    """
    if holding_cost_per_unit <= 0:
        return None
    eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)
    return round(eoq, 1)


def build_inventory_plan(daily_demand_df, category_name, lead_time_days=7,
                          service_level=0.95, order_cost=50, holding_cost_per_unit=2):
    """
    Full inventory plan for one category, using its historical demand stats.
    """
    cat_data = daily_demand_df[daily_demand_df['Category Name'] == category_name]['quantity']

    if len(cat_data) < 10:
        return {"error": f"Not enough data for category '{category_name}'"}

    avg_daily_demand = cat_data.mean()
    demand_std = cat_data.std()
    annual_demand = avg_daily_demand * 365

    safety_stock = calculate_safety_stock(demand_std, lead_time_days, service_level)
    reorder_point = calculate_reorder_point(avg_daily_demand, lead_time_days, safety_stock)
    eoq = calculate_eoq(annual_demand, order_cost, holding_cost_per_unit)

    return {
        "category": category_name,
        "avg_daily_demand": round(avg_daily_demand, 1),
        "demand_std": round(demand_std, 1),
        "lead_time_days": lead_time_days,
        "service_level": service_level,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "economic_order_quantity": eoq,
        "annual_demand_estimate": round(annual_demand, 0)
    }


if __name__ == "__main__":
    daily = pd.read_csv('../data/daily_demand_by_category.csv', parse_dates=['order_date'])

    # Build a plan for each of your top categories
    top_categories = daily.groupby('Category Name')['quantity'].sum().sort_values(ascending=False).head(15).index

    all_plans = []
    for cat in top_categories:
        plan = build_inventory_plan(daily, cat, lead_time_days=7, service_level=0.95)
        all_plans.append(plan)
        print(f"\n{cat}:")
        print(f"  Avg daily demand: {plan['avg_daily_demand']}")
        print(f"  Safety stock: {plan['safety_stock']} units")
        print(f"  Reorder point: {plan['reorder_point']} units")
        print(f"  EOQ (order quantity): {plan['economic_order_quantity']} units")

    results_df = pd.DataFrame(all_plans)
    results_df.to_csv('../outputs/inventory_plan.csv', index=False)
    print("\nSaved full inventory plan to outputs/inventory_plan.csv")