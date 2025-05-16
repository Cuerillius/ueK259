from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn import metrics

data, labels = make_blobs(n_samples=50, centers=4, cluster_std=2.75, random_state=42)


experienceData, testData, experienceLabels, testLabels = train_test_split(
    data, labels, test_size=0.20
)


def k_nearest_neighbour(newPoint, experienceData, experienceLabels, numberOfNeighbours):

    distances = []
    for i, coord in enumerate(experienceData):
        distances.append([euclidean_distances([newPoint], [coord])[0][0], i])
    distances = np.array(distances)
    # nearest_neighbours = np.sort(distances[:, 0])[:numberOfNeighbours]
    nearest_neighbours = np.array(
        sorted(distances, key=lambda x: x[0])[:numberOfNeighbours]
    )

    labels = []
    for nearest_neighbour in nearest_neighbours[:, 1]:
        labels.append(experienceLabels[int(nearest_neighbour)])
    return max(set(labels), key=labels.count)


calculatet_labels = []
for i, coord in enumerate(testData):
    calculatet_labels.append(
        k_nearest_neighbour(coord, experienceData, experienceLabels, 5)
    )

results = []
for i, label in enumerate(calculatet_labels):
    results.append(label == testLabels[i])

print(100 / len(results) * np.sum(results), "% Success")

cmap = colors.ListedColormap(["red", "green"])

fig, ax = plt.subplots()
ax.scatter(x=experienceData[:, 0], y=experienceData[:, 1], c=experienceLabels, s=5)
ax.scatter(x=testData[:, 0], y=testData[:, 1], c=results, s=20, marker="s", cmap=cmap)
plt.plot()

plt.show()
