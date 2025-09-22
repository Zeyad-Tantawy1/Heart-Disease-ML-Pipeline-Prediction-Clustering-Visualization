import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def save_pipeline(model, X, y, path="models/final_pipeline.pkl"):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, path)
    print(f"✅ Final pipeline saved as {path}")
