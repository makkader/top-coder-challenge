import json
from xgboost import XGBRegressor
import pandas as pd
from modeling import get_trained_model, feature_improvement

def read_test_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def generate_output(predictions):
    with open("private_results.txt", "w") as f:
        for pred in predictions:
            f.write(str(pred) + "\n")

if __name__ == "__main__":
    model = get_trained_model()

    df = read_test_data("private_cases.json")
    X = feature_improvement(df)
    predictions = model.predict(X)
    generate_output(predictions.round(2))