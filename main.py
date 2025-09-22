from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.eda import run_eda
from src.pca_module import apply_pca
from src.feature_selection import feature_importance_rf, rfe_selection, chi2_selection
from src.supervised_models import evaluate_models
from src.tuning import tune_rf, tune_lr, tune_svm
from src.clustering import kmeans_clustering, hierarchical_clustering
from src.utils import save_pipeline

def main():
    df_raw = load_raw_data("data/heart+disease.zip")
    print("Raw data shape:", df_raw.shape)

    print("\n=== Running EDA on raw data ===")
    run_eda(df_raw)

    df_ready = preprocess_data(df_raw)
    print("\nData after preprocessing:", df_ready.shape)

    df_pca = apply_pca(df_ready)

    X = df_ready.drop("target", axis=1)
    y = df_ready["target"]

    print("\n=== Feature Importance (Random Forest) ===")
    feat_imp = feature_importance_rf(X, y)
    print(feat_imp.head())

    print("\n=== RFE Selection ===")
    print(rfe_selection(X, y))

    print("\n=== Chi-Square Selection ===")
    print(chi2_selection(X, y))

    print("\n=== Evaluating Supervised Models ===")
    results = evaluate_models(df_ready)
    print(results)

    print("\n=== Hyperparameter Tuning ===")
    rf_best = tune_rf(X, y)
    lr_best = tune_lr(X, y)
    svm_best = tune_svm(X, y)
    print("Best RF:", rf_best)
    print("Best LR:", lr_best)
    print("Best SVM:", svm_best)

    print("\n=== Clustering ===")
    kmeans_clustering(X, y)
    hierarchical_clustering(X, y)

    print("\n=== Saving Final Model Pipeline ===")
    save_pipeline(rf_best, X, y, path="models/final_pipeline.pkl")

if __name__ == "__main__":
    main()
