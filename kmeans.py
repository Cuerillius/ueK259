from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
import numpy as np
import math

features, true_labels = make_blobs(
    n_samples=500, centers=4, cluster_std=1.75, random_state=43
)


def visualise(features, estimated_labels, cluster_centers):
    fig, ax = plt.subplots()
    ax.scatter(x=features[:, 0], y=features[:, 1], c=estimated_labels, s=5)
    ax.scatter(x=cluster_centers[:, 0], y=cluster_centers[:, 1], s=50, c="black")
    plt.plot()


def k_means(featureArray: np.ndarray, nGroups):

    estimated_labels = np.array(list(range(len(featureArray))))
    cluster_centers = np.random.rand(nGroups, 2) * 2
    iterations = 0
    while iterations < 10000:
        prev_cluster_centers = cluster_centers.copy()
        for i, coord in enumerate(featureArray):
            temp_center_distances = []
            for center_start_coord in cluster_centers:
                temp_center_distances.append(
                    math.sqrt(
                        ((coord[0] - center_start_coord[0]) ** 2)
                        + ((coord[1] - center_start_coord[1]) ** 2)
                    )
                )
            estimated_labels[i] = temp_center_distances.index(
                min(temp_center_distances)
            )

        for i in range(nGroups):
            if np.any(estimated_labels == i):
                cluster_centers[i] = featureArray[estimated_labels == i].mean(axis=0)

        iterations += 1
        if (prev_cluster_centers == cluster_centers).all():
            break
    return estimated_labels, cluster_centers


estimated_labels, cluster_centers = k_means(features, 4)

visualise(features, estimated_labels, cluster_centers)


def evaluate(featureArray, estimated_labels, cluster_centers):
    sum = 0
    count = 0
    for i, center in enumerate(cluster_centers):
        if np.any(estimated_labels == i):
            for coord in featureArray[estimated_labels == i]:
                sum += abs(math.hypot((coord[0] - center[0]), (coord[1] - coord[1])))
                count += 1

    return sum / count


evaluate(features, estimated_labels, cluster_centers)

evaluation = []
k_times = 10
for k in range(k_times):
    estimated_labels, cluster_centers = k_means(features, k + 1)

    evaluation.append(evaluate(features, estimated_labels, cluster_centers))
plt.figure(), plt.plot(range(k_times), evaluation)
plt.show()
