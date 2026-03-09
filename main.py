from typing import cast, List
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def pca_extract(data: pd.DataFrame):

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    return pd.DataFrame(data=components, columns=["PC1", "PC2"])


def analysis(df: pd.DataFrame):
    inertia = []

    K = range(2, 16)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, "bx-")
    plt.xlabel("# Clusters")
    plt.ylabel("Inertia")
    plt.title(f"Elbow - {df.columns[0]} vs {df.columns[1]}")
    plt.show()
    plt.savefig(f"images/elbow-{df.columns[0]}-vs-{df.columns[1]}")


def clusters_get(df: pd.DataFrame, k: int):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = pd.DataFrame(kmeans.fit_predict(df), index=df.index, columns=["cluster"])

    return pd.DataFrame(clusters)


def draw_results(dataframes: List[pd.DataFrame]):
    seaborn.set_theme(style="whitegrid")
    _, axes = plt.subplots(1, 2, figsize=(18, 5))

    for i, target in enumerate(targets):
        partial = df[target]
        cluster = clusters_get(cast(pd.DataFrame, partial), 5)
        dataframes.append(pd.concat([partial, cluster], axis=1))

        seaborn.scatterplot(
            ax=axes[i],
            data=dataframes[i],
            x=target[0],
            y=target[1],
            hue=dataframes[i]["cluster"],
            palette="viridis",
            legend="full",
        )
        axes[i].set_title(f"{target[0]} vs {target[1]}")
        axes[i].set_xlabel(target[0])
        axes[i].set_ylabel([target[1]])

    plt.tight_layout()
    plt.show()
    plt.savefig("images/clusters.png")


df = pd.read_csv("files/Country-data.xls")
countries = df.pop("country")


target_one = ["child_mort", "gdpp"]
target_two = ["life_expec", "gdpp"]

targets = [target_one, target_two]
dataframes = []

for i, target in enumerate(targets):
    partial = df[target]
    cluster = clusters_get(cast(pd.DataFrame, partial), 5)
    dataframes.append(pd.concat([partial, cluster], axis=1))

# analysis(dataframes[0]) #optimal between 4 and 6
# analysis(dataframes[1]) #optimal between 4 and 6

draw_results(dataframes)
