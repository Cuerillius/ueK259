from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
import numpy as np
import math

features, true_labels = make_blobs(
    n_samples=500, centers=4, cluster_std=0.75, random_state=42
)

""" fig, ax = plt.subplots()
ax.scatter(x=features[:, 0], y=features[:, 1], s=5, c=true_labels)
 """

""" cluster_centers = np.array([[-7.5, -7.5], [-8.5, 7.5], [-2.5, 10], [5, 2.5]])
features = np.array(
    [
        [-7.5, -7.5],
        [-8.5, 7.5],
        [-2.5, 10],
        [5, 2.5],
        [-8.5, -8.5],
        [-9.5, 6.5],
        [-3.5, 9],
        [4, 1.5],
        [-6.5, -6.5],
        [-7.5, 8.5],
        [-1.5, 11],
        [6, 3.5],
    ]
)
estimated_labels = np.array(
    [
        2,
        1,
        3,
        4,
        2,
        1,
        3,
        4,
        2,
        1,
        3,
        4,
    ]
)
 """


def visualise(features, estimated_labels, cluster_centers):
    fig, ax = plt.subplots()
    ax.scatter(x=features[:, 0], y=features[:, 1], c=estimated_labels, s=5)
    ax.scatter(x=cluster_centers[:, 0], y=cluster_centers[:, 1], s=50, c="black")
    plt.plot()


def k_means(featureArray: np.ndarray, nGroups):

    estimated_labels = np.array(list(range(len(featureArray))))
    cluster_centers = np.random.rand(nGroups, 2) * 2
    iterations = 0
    while iterations < 10:
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
        visualise(features, estimated_labels, cluster_centers)
        plt.show()
    return estimated_labels, cluster_centers


estimated_labels, cluster_centers = k_means(features, 4)
