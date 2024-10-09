# src/mymodelapi/predictions/model.py
import pickle

model_path = "C:/Users/teeyob/CreditRiskAnalyzer/notebooks/random_forest_model.pkl"  # Adjust the path as necessary

with open(model_path, "rb") as f:
    model = pickle.load(f)
