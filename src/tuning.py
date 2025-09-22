from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def tune_rf(X, y):
    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_params, n_iter=10, cv=5, scoring="f1", random_state=42)
    search.fit(X, y)
    return search.best_estimator_

def tune_lr(X, y):
    lr_params = {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs"]}
    search = GridSearchCV(LogisticRegression(max_iter=5000), lr_params, cv=5, scoring="f1")
    search.fit(X, y)
    return search.best_estimator_

def tune_svm(X, y):
    svm_params = {"C": [0.1, 1, 10], "kernel": ["linear","rbf"], "gamma": ["scale","auto"]}
    search = GridSearchCV(SVC(probability=True), svm_params, cv=5, scoring="f1")
    search.fit(X, y)
    return search.best_estimator_
