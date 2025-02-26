import numpy as np
from scipy.spatial.distance import mahalanobis

def mahalanobis_distance(poly1, poly2):
    c1, c2 = np.mean(poly1, axis=0), np.mean(poly2, axis=0)
    cov = np.cov(np.vstack((poly1, poly2)).T)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        raise ValueError("Singular covariance matrix. Ensure polygons are not collinear.")
    return mahalanobis(c1, c2, inv_cov)

def get_polygon(n):
    return [tuple(map(float, p.split())) for p in input(f"Polygon {n} (x y, ...): ").split(',')]

try:
    print("Mahalanobis Distance:", mahalanobis_distance(get_polygon(1), get_polygon(2)))
except Exception as e:
    print("Error:", e)