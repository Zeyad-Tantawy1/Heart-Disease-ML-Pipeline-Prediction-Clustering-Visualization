import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

def kmeans_clustering(X, y, max_k=10):
    inertia = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1,max_k+1), inertia, marker="o")
    plt.xlabel("K"); plt.ylabel("Inertia")
    plt.title("Elbow Method - KMeans")
    plt.show()

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    ari = adjusted_rand_score(y, clusters)
    print("ARI (KMeans vs Target):", ari)
    return clusters

def hierarchical_clustering(X, y):
    linked = linkage(X, method="ward")
    plt.figure(figsize=(10,6))
    dendrogram(linked, truncate_mode="level", p=5)
    plt.title("Hierarchical Dendrogram")
    plt.show()

    hc = AgglomerativeClustering(n_clusters=2, linkage="ward")
    clusters = hc.fit_predict(X)
    ari = adjusted_rand_score(y, clusters)
    print("ARI (Hierarchical vs Target):", ari)
    return clusters
