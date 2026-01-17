import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    df = pd.read_csv("congestion_dataset.csv")

    features = ["nSta", "nBg", "offeredMbps", "throughputMbps", "meanDelayMs", "lossRate"]
    X = df[features]
    y = df["congestionLabel"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred))

    joblib.dump(model, "congestion_model.joblib")
    print("✅ Saved model -> congestion_model.joblib")

if __name__ == "__main__":
    main()
