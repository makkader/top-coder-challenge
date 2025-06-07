import xgboost as xgb
import pandas as pd
import numpy as np
import json

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    reshaped_data = []
    for example in data:
        rs = {
            'trip_duration_days':example.get("input").get("trip_duration_days"),
            'miles_traveled':example.get("input").get("miles_traveled"),
            'total_receipts_amount':example.get("input").get("total_receipts_amount"),
            "expected_output": example.get("expected_output") 
        }
        reshaped_data.append(rs)
    
    return pd.DataFrame(reshaped_data)


def feature_improvement(X):
    # 1. Per Diem Related Features:
    # Base Per Diem
    X["per_diem_base"] = X["trip_duration_days"] * 100
    
    # Is 5-day trip
    X["is_5_day_trip"] = (X["trip_duration_days"] == 5).astype(int)
    
    # Is long trip (e.g., >= 8 days)
    X["is_long_trip"] = (X["trip_duration_days"] >= 8).astype(int)
    
    # Trip duration squared
    X["trip_duration_squared"] = X["trip_duration_days"]**2
    
    # 2. Mileage Related Features:
    # Miles per day
    X["miles_per_day"] = X["miles_traveled"] / X["trip_duration_days"]
    
    # Miles per day squared
    X["miles_per_day_squared"] = X["miles_per_day"]**2
    
    # Mileage Tier 1 (first 100 miles)
    MILEAGE_TIER_1_THRESHOLD = 100
    X["mileage_tier_1"] = np.minimum(X["miles_traveled"], MILEAGE_TIER_1_THRESHOLD)
    
    # Mileage Tier 2 (miles beyond first 100)
    X["mileage_tier_2"] = np.maximum(0, X["miles_traveled"] - MILEAGE_TIER_1_THRESHOLD)
    
    # Log miles traveled (add 1 to avoid log(0) if miles_traveled can be 0)
    X["log_miles_traveled"] = np.log(X["miles_traveled"] + 1)
    
    # 3. Receipts Related Features:
    # Receipts per day
    X["receipts_per_day"] = X["total_receipts_amount"] / X["trip_duration_days"]
    
    # Receipts per day squared
    X["receipts_per_day_squared"] = X["receipts_per_day"]**2
    
    # Receipts low penalty (e.g., < $50)
    RECEIPTS_LOW_THRESHOLD = 50
    X["receipts_low_penalty"] = (X["total_receipts_amount"] < RECEIPTS_LOW_THRESHOLD).astype(int)
    
    # Receipts high penalty (e.g., > $1000)
    RECEIPTS_HIGH_THRESHOLD = 1000
    X["receipts_high_penalty"] = (X["total_receipts_amount"] > RECEIPTS_HIGH_THRESHOLD).astype(int)
    
    # Receipts cents .49 or .99 (rounding bug)
    X["receipts_cents"] = (X["total_receipts_amount"] * 100).astype(int) % 100
    
    X["receipts_cents_49_99"] = ((X["receipts_cents"] == 49) | (X["receipts_cents"] == 99)).astype(int)
    
    # Log total receipts amount (add 1 to avoid log(0) if total_receipts_amount can be 0)
    X["log_total_receipts_amount"] = np.log(X["total_receipts_amount"] + 1)
    
    # 4. Interaction Features:
    # Duration x Miles
    X["duration_x_miles"] = X["trip_duration_days"] * X["miles_traveled"]
    
    # Duration x Receipts
    X["duration_x_receipts"] = X["trip_duration_days"] * X["total_receipts_amount"]
    
    # Miles x Receipts
    X["miles_x_receipts"] = X["miles_traveled"] * X["total_receipts_amount"]
    
    # Efficiency x Spending (miles_per_day * receipts_per_day)
    X["efficiency_x_spending"] = X["miles_per_day"] * X["receipts_per_day"]
    return X

def train_model_xgb(df):
    X = df.drop(columns=['expected_output'])
    y = df['expected_output']
    X = feature_improvement(X)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, subsample=0.6, n_estimators=500, min_child_weight=3, max_depth=3, learning_rate=0.05, gamma=0, colsample_bytree=0.8)
    xgb_model.fit(X, y)
    return xgb_model



def get_trained_model():
    df = read_data("public_cases.json")
    model = train_model_xgb(df)
    return model