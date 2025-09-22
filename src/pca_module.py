import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def apply_pca(df: pd.DataFrame, variance_threshold: float = 0.95):
    X = df.drop("target", axis=1)
    y = df["target"]

    pca = PCA()
    X_pca = pca.fit_transform(X)

    explained_var = pca.explained_variance_ratio_
    cum_var = explained_var.cumsum()

    plt.plot(range(1, len(cum_var)+1), cum_var, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by PCA")
    plt.show()

    plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", alpha=0.7)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA Scatter Plot")
    plt.colorbar(label="Target")
    plt.show()

    n_components = (cum_var >= variance_threshold).argmax() + 1
    print(f"Number of components for {variance_threshold*100}% variance: {n_components}")

    pca_final = PCA(n_components=n_components)
    X_pca_final = pca_final.fit_transform(X)

    df_pca = pd.DataFrame(X_pca_final, columns=[f"PC{i+1}" for i in range(n_components)])
    df_pca["target"] = y.reset_index(drop=True)

    return df_pca
