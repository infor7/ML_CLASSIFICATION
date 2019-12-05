import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations


def two_lines_intersection(a, b, c, d):
    x = (d-b)/(a-c)
    return x, a*x +b

def line_from_two_points(a, b):
    x_delta = b[0] - a[0]
    y_delta = b[1] - a[1]
    tan = y_delta/x_delta
    y_zero = a[1]-a[0]*tan
    return [tan, y_zero]

def split_by_class(dataset, labels):
    sorted_indices = np.argsort(labels)
    dataset = dataset[sorted_indices]
    labels = labels[sorted_indices]
    dataset_classed = np.split(dataset, np.where(labels[:-1] != labels[1:])[0] + 1)
    return dataset_classed

def intersections_of_hulls(vertices1, vertices2):
    for index in range(len(vertices1)):
        for j in range(len(vertices2)):
            return


def all_intersections(hulls):
    for comb in combinations(range(len(hulls)), 2):
        hull1 = hulls[comb[0]]
        hull2 = hulls[comb[1]]


iris = load_iris()
labels = iris.target
data = iris.data
coord1 = 0
coord2 = 1

dataset_classed = split_by_class(data, labels)
separation_lines = []
for perm in combinations(range(len(dataset_classed)), 2):
    mass_center1 = np.mean(dataset_classed[perm[0]], axis=0)
    mass_center2 = np.mean(dataset_classed[perm[1]], axis=0)
    mass_center = (mass_center1 + mass_center2)/2
    z0 = np.polyfit(dataset_classed[perm[0]][:, coord1], dataset_classed[perm[0]][:, coord2], 1)
    z1 = np.polyfit(dataset_classed[perm[1]][:, coord1], dataset_classed[perm[1]][:, coord2], 1)
    second_point = two_lines_intersection(z0[0], z0[1], z1[0], z1[1])
    line_separation = line_from_two_points(mass_center[coord1:coord2+1], second_point)
    # if perm == (1,2):
    plt.plot([2, 5], [line_separation[0] * 2 + line_separation[1], line_separation[0] * 5 + line_separation[1]], 'b--')
    separation_lines.append(line_separation)

hulls = []
for index, clas in enumerate(dataset_classed):
    hull = ConvexHull(clas[:,coord1:coord2+1])
    hulls.append(hull)
    plt.plot(clas[:,coord1], clas[:, coord2], '.')
    plt.plot(clas[hull.vertices,coord1], clas[hull.vertices,coord2], 'r-')



plt.show()
