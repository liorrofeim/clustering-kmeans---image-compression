import numpy as np


def get_random_centroids(X, k):
    """
    Each centroid is a point in RGB space (color) in the image.
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids.
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array.
    """
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float)


def lp_distance(X, centroids, p=2):
    """
    Inputs:
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of
    all points in RGB space from all centroids
    """
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    X_exp = np.expand_dims(X, axis=0)  # Shape becomes (1, num_pixels, 3)
    centroids_exp = np.expand_dims(centroids, axis=1)  # Shape becomes (k, 1, 3)

    # Now X_exp and centroids_exp can be broadcasted to the shape (k, num_pixels, 3)
    distances = np.sum(np.abs(X_exp - centroids_exp) ** p, axis=2) ** (1 / p)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances


def kmeans(X, k, p, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for _ in range(max_iter):
        old_centroids = centroids.copy()
        distances = lp_distance(X, centroids, p)

        # Step 2: Assign each data point to the closest centroid
        classes = np.argmin(distances, axis=0)

        # Step 3: Calculate new centroids
        for i in range(k):
            centroids[i] = np.mean(X[classes == i], axis=0)

        # Check for convergence (i.e., no change in centroids)
        if np.all(old_centroids == centroids):
            break
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes


def initialize_centroids(X, k):
    num_pixels, _ = X.shape

    # Initialize the centroids container
    centroids = np.zeros((k, 3))

    # Step 1: Choose a centroid uniformly at random among the data points
    idx = np.random.randint(num_pixels)
    centroids[0] = X[idx]

    for i in range(1, k):
        # Vectorized computation of distance
        dist = np.sqrt(((X - centroids[:i, np.newaxis]) ** 2).sum(axis=2))
        min_dist = dist.min(axis=0)

        # Squared distances
        sq_dist = min_dist**2

        # Probability distribution
        total_sq_dist = sq_dist.sum()
        probabilities = sq_dist / total_sq_dist

        # Cumulative probabilities
        cumulative_probabilities = np.cumsum(probabilities)

        # Randomly select the next centroid
        rand_val = np.random.rand()
        for j, cp in enumerate(cumulative_probabilities):
            if rand_val < cp:
                idx_next = j
                break

        centroids[i] = X[idx_next]

    return centroids


def kmeans_pp(X, k, p, max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    print("start")
    centroids = initialize_centroids(X, k)
    print("finish stage 1")
    for _ in range(max_iter):
        old_centroids = centroids.copy()
        distances = lp_distance(X, centroids, p)

        # Step 2: Assign each data point to the closest centroid
        classes = np.argmin(distances, axis=0)

        # Step 3: Calculate new centroids
        for i in range(k):
            centroids[i] = np.mean(X[classes == i], axis=0)

        # Check for convergence (i.e., no change in centroids)
        if np.all(old_centroids == centroids):
            break

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes
