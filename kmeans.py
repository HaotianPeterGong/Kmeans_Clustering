import numpy as np
from numpy.random import default_rng

def kmeans(X:np.ndarray, k:int, random_state=7, 
           centroids=None, max_iter=30, tolerance=1e-2, return_distance=False,
          show_steps=False):
    """
    centroids is an k x p matrix containing the k centroids (vectors are p long). 
    The labels return value is a list of length n the label associated with each input vector. 
    I use the tolerance as a general guideline for comparing previous and next generation centroids. 
    If the average norm of centroids-previous_centroids is less than the tolerance, I stop. 
    I also have max_iter as a failsafe to prevent it from going too long. 
    By default, centroids=None indicates that your algorithm should randomly select k unique centroids.
    """
    rng = default_rng(seed=random_state)        # select k unique points from X as initial centroids
    if centroids == 'kmeans++':
        centroids = select_centroids(X,k,random_state)
    else:
        centroids = rng.choice(X, size=k, replace=False) # replace=False to get the unique samples

    for i in range(1, max_iter+1): # 0 is position for initial centroid
        centroids_prev = centroids
        clusters = np.array([])
        for x in X:
            j = np.sum(np.square(x-centroids), axis=1).argmin()
            clusters = np.append(clusters, j) # find closest centroid of each x
        centroid_list = []
        for cluster in range(k):
            cluster_idx = np.where(clusters==cluster)[0]
            centroid = X[cluster_idx].mean(axis=0).tolist()
            centroid_list.append(centroid)
        centroids = np.array(centroid_list) # get new centroids in the next step
        if show_steps:
            print(i)
        if np.linalg.norm(centroids - centroids_prev) < tolerance:
            break
            
    labels = np.array([int(l) for l in clusters]) # change float type to int
    
    if return_distance:
        distance = 0
        for x in X:
            distance_matrix = np.sqrt(np.sum(np.square(x-centroids), axis=1)) # distance_matrix is a 1*k matrix
            distance = np.sum(distance_matrix.min()) # reduce distance to 1*1
            distance += distance
        return centroids, labels, distance
    return centroids, labels 


def select_centroids(X, k, random_state=7):
    """
    kmeans++ algorithm to select initial points:

    1. Pick first point randomly
    2. Pick next k-1 points by selecting points that maximize the minimum
       distance to all existing clusters. So for each point, compute distance
       to each cluster and find that minimum.  Among the min distances to a cluster
       for each point, find the max distance. The associated point is the new centroid.

    Return centroids as k x p array of points from X.
    """
    rng = default_rng(seed=random_state)
    centroids = rng.choice(X, size=1, replace=False) # replace=False to get one unique samples

    for i in range(1, k):
        distance_array = np.array([])
        for x in X:
            point_cluster_min_dis = np.sum(np.square(x-centroids), axis=1).min()
            distance_array = np.append(distance_array, point_cluster_min_dis)
        centroid = [X[distance_array.argmax()].tolist()] # create same shape as initial centroid [[]]
        centroids = np.append(centroids, centroid, axis=0) # np.append([[]], [[]], axis=0)
    
    return centroids
