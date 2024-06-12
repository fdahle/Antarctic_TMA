"""get the optimal number of clusters for tie points"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def determine_optimal_clusters(tps: np.ndarray, max_k: int = 50, top_n_changes: int = 3) -> int:
    """
    Determines the optimal number of clusters for tie points using the Elbow method and silhouette scores.

    Args:
        tps (np.ndarray): 2D numpy array of tie points.
        max_k (int, optional): Maximum number of clusters to consider. Defaults to 50.
        top_n_changes (int, optional): Number of top changes in distortions to consider for determining the elbow point.
            Defaults to 3.

    Returns:
        int: The optimal number of clusters.
    """
    distortions = []
    sil_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(tps[:, :2])
        distortions.append(kmeans.inertia_)  # noqa
        sil_scores.append(silhouette_score(tps[:, :2], kmeans.labels_))  # noqa

    # Elbow method
    changes = [distortions[i] - distortions[i + 1] for i in range(len(distortions) - 1)]
    largest_changes_indices = np.argsort(changes)[-top_n_changes:]
    elbow_point = largest_changes_indices[-1] + 2  # You may adjust how you pick from the top changes

    best_silhouette = np.argmax(sil_scores) + 2

    return max(elbow_point, best_silhouette)
