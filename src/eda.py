import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_eda(df: pd.DataFrame):
    print("Missing values per column:\n", df.isna().sum())

    # Histograms
    df.hist(figsize=(15,12), bins=20)
    plt.suptitle("Histograms of All Features", fontsize=16)
    plt.show()

    # Boxplots
    plt.figure(figsize=(15,8))
    sns.boxplot(data=df.drop(columns=["target"], errors="ignore"))
    plt.xticks(rotation=45)
    plt.title("Boxplots for Features")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # Target Distribution
    if "target" in df.columns:
        sns.countplot(x="target", data=df)
        plt.title("Target Distribution")
        plt.show()
