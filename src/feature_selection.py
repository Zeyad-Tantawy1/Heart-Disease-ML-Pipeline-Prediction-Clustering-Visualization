import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

def feature_importance_rf(X, y):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
    feat_imp = feat_imp.sort_values("Importance", ascending=False)

    sns.barplot(x="Importance", y="Feature", data=feat_imp)
    plt.title("Feature Importance (Random Forest)")
    plt.show()

    return feat_imp

def rfe_selection(X, y, n_features=5):
    lr = LogisticRegression(max_iter=2000)
    rfe = RFE(lr, n_features_to_select=n_features)
    rfe.fit(X, y)
    return X.columns[rfe.support_].tolist()

def chi2_selection(X, y, k=5):
    X_scaled = MinMaxScaler().fit_transform(X)
    chi2_selector = SelectKBest(chi2, k=k)
    chi2_selector.fit(X_scaled, y)
    return X.columns[chi2_selector.get_support()].tolist()
