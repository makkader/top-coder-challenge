import sys
from modeling import get_trained_model
import pandas as pd
import numpy as np



def add_trip_features(df):
    # 1. Per Diem Related Features:
    df["per_diem_base"] = df["trip_duration_days"] * 100
    df["is_5_day_trip"] = (df["trip_duration_days"] == 5).astype(int)
    df["is_long_trip"] = (df["trip_duration_days"] >= 8).astype(int)
    df["trip_duration_squared"] = df["trip_duration_days"]**2

    # 2. Mileage Related Features:
    df["miles_per_day"] = df["miles_traveled"] / df["trip_duration_days"]
    df["miles_per_day_squared"] = df["miles_per_day"]**2

    MILEAGE_TIER_1_THRESHOLD = 100
    df["mileage_tier_1"] = np.minimum(df["miles_traveled"], MILEAGE_TIER_1_THRESHOLD)
    df["mileage_tier_2"] = np.maximum(0, df["miles_traveled"] - MILEAGE_TIER_1_THRESHOLD)
    df["log_miles_traveled"] = np.log(df["miles_traveled"] + 1)

    # 3. Receipts Related Features:
    df["receipts_per_day"] = df["total_receipts_amount"] / df["trip_duration_days"]
    df["receipts_per_day_squared"] = df["receipts_per_day"]**2

    RECEIPTS_LOW_THRESHOLD = 50
    df["receipts_low_penalty"] = (df["total_receipts_amount"] < RECEIPTS_LOW_THRESHOLD).astype(int)

    RECEIPTS_HIGH_THRESHOLD = 1000
    df["receipts_high_penalty"] = (df["total_receipts_amount"] > RECEIPTS_HIGH_THRESHOLD).astype(int)

    df["receipts_cents"] = (df["total_receipts_amount"] * 100).astype(int) % 100
    df["receipts_cents_49_99"] = ((df["receipts_cents"] == 49) | (df["receipts_cents"] == 99)).astype(int)

    df["log_total_receipts_amount"] = np.log(df["total_receipts_amount"] + 1)

    # 4. Interaction Features:
    df["duration_x_miles"] = df["trip_duration_days"] * df["miles_traveled"]
    df["duration_x_receipts"] = df["trip_duration_days"] * df["total_receipts_amount"]
    df["miles_x_receipts"] = df["miles_traveled"] * df["total_receipts_amount"]
    df["efficiency_x_spending"] = df["miles_per_day"] * df["receipts_per_day"]

    return df


if __name__ == "__main__":
    model = get_trained_model()
    try:
        num1 = float(sys.argv[1])
        num2 = float(sys.argv[2])
        num3 = float(sys.argv[3])

        tdf=pd.DataFrame([[num1, num2, num3]], columns=['trip_duration_days', 'miles_traveled', 'total_receipts_amount'])
        tdf = add_trip_features(tdf)
        prediction = model.predict(tdf)
        reimbursement = prediction[0].round(2)
        print(reimbursement)

    except IndexError:
        print("Error: Please provide three arguments.")
        sys.exit(1)
    except ValueError:
        print("Error: All arguments must be valid numbers.")
        sys.exit(1)
