import numpy as np
import pandas as pd

class KMeans:
    """
    In this class, we want to implement K-means clustering algorithm from scratch :)
    """
    def __init__(self) -> None:
        self.n_cluster_ = None
        self.centroids_ = None


    def fit(self, X: pd.DataFrame, n_cluster, n_iter=100, max_iter=100) -> None:
        self.n_cluster_ = n_cluster
        self.centroids_ = None
        X = X.values

        min_cost = np.inf
        for i in range(max_iter):
            centroid_coordinates, centroid_indices = self.run_KMeans(X, n_cluster, n_iter)
            cost = self.compute_cost(X, centroid_indices, centroid_coordinates)
            if cost < min_cost: # We found a better model
                min_cost = cost
                self.centroids_ = centroid_coordinates


    def fit_predict(self, X: pd.DataFrame, n_cluster, n_iter=100, max_iter=100) -> np.array:
        self.fit(X, n_cluster, n_iter, max_iter)
        return self.predict(X)


    def predict(self, X: pd.DataFrame):
        X = X.values
        m = X.shape[0]
        n = self.n_cluster_
        clusters = np.zeros(m, dtype=int)

        for i in range(m):
            distances = np.zeros(n)
            for j in range(n):
                distances[j] = np.linalg.norm(X[i] - self.centroids_[j])
            clusters[i] = np.argmin(distances)
            del distances
        
        return clusters
     

    def run_KMeans(self, X: np.array, n_clusters, n_iter) -> None:
        
        # First we have to randomly initialize the centroids for for our clusters
        X_perm = np.random.permutation(X)
        # Then we choose first K points as our centroids
        init_centoids = X_perm[:n_clusters]
        
        for i in range(n_iter):
            centroid_indices = self.find_closest_centroids(X, init_centoids)
            init_centoids = self.compute_centroids(X, centroid_indices)            

        return init_centoids, centroid_indices


    def compute_cost(self, X: np.array, centroid_indices: np.array, centroid_coordinates: np.array):
        m = X.shape[0]
        cost = .0
        for i in range(m):
            cost += np.linalg.norm(X[i] - centroid_coordinates[centroid_indices[i]])
        
        return cost


    def compute_centroids(self, X: np.array, centroids: np.array) -> np.array:
        """
        This method gets the X and centroids as input, and returns the new centroids for each cluster.
        centroids in this method is a numpy array which contains m indices, the number of clusters for each point in X.
        """    
        n = X.shape[1]
        centroid_coordinates = np.zeros((self.n_cluster_, n))
        # Iterating over the points in each cluster
        for i in range(self.n_cluster_):
            points = X[centroids == i]
            mean = np.mean(points, axis=0)
            centroid_coordinates[i] = mean

        return centroid_coordinates
    
    
    def find_closest_centroids(self, X: np.array, centroids: np.array):
        """
        This method gets X and centroids as inputs, and determins each point belongs to which cluster.
        Centroids in this method is a numpy array which has the sahpe(K, n), where K is the number of clusters and n is the number 
        of features.
        """
        m = X.shape[0]
        n = self.n_cluster_
        centroids_indices = np.zeros(m, dtype=int)
        
        for i in range(m):
            distances = np.zeros(n)
            for j in range(n):
                distances[j] = np.linalg.norm(X[i] - centroids[j])
            centroids_indices[i] = np.argmin(distances)
            del distances 
        return centroids_indices