import numpy as np

def determine_optimal_clusters(tps, max_k=50, top_n_changes=3):
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
