import numpy as np

def find_nearest_edge_point(point, edge_points):
    edge_points = edge_points.reshape(-1, 2)
    # Calculate the Euclidean distances
    distances = np.linalg.norm(edge_points - point, axis=1)

    if len(distances) == 0:
        raise ValueError("No points to calculate distance from.")
    nearest_index = np.argmin(distances)

    return edge_points[nearest_index]