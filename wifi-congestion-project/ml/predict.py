import joblib
import pandas as pd

LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

def main():
    model = joblib.load("congestion_model.joblib")

    sample = pd.DataFrame([{
        "nSta": 20,
        "nBg": 10,
        "offeredMbps": 20,
        "throughputMbps": 8.0,
        "meanDelayMs": 45.0,
        "lossRate": 0.12
    }])

    pred = int(model.predict(sample)[0])
    print("Predicted congestion:", LABELS[pred])

if __name__ == "__main__":
    main()
