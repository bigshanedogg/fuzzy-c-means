import numpy as np

class fuzzy_cmeans:
    """
    The extension version for fuzzy_cmenas with various distance metrics including Euclidean, Cosine, Mahalanobis.
    Implementation of James Bezdek, "FCM: The fuzzy c-means clustering algorithm"
    https://www.sciencedirect.com/science/article/pii/0098300484900207
    
    How to use
    Define fuzzy_cmeans object with parameters; n_clusters, p: factors that determine the probability of belonging to a group, metric: distance metric
    """
    
    n_clusters = None
    m = None # fuzziness
    metric = None
    weights = None
    centers = None
    X = None
    
    # iter terminate condition 
    min_error_change = None
    min_weights_change = None
    max_iter = None
    verbose = None
    _underflow = 1e-6
    
    def __init__(self, n_clusters, m, metric="euclidean",
                 min_error_change=1e-4, min_weights_change=1e-6, max_iter=1000, 
                 verbose=False):
        self.n_clusters = n_clusters
        self.m = m
        self.metric = metric
        self.verbose = verbose
        self.min_error_change = min_error_change
        self.min_weights_change = min_weights_change
        self.max_iter = max_iter
        
    def fit(self, X, optimization=["weights"]):
        X = np.array(X)
        self.X = X
        
        # 1. weight matrix & centers initialization
        if self.verbose: print("Weight matrix & Centers initialization")
        self.weights = self._init_weights_rand(n_points=len(X), n_clusters=self.n_clusters, _underflow=self._underflow)

        ## iterate
        # first iteration
        if self.verbose: print("iter {n_iter}:".format(n_iter=0))
        if self.verbose: print("\tCluster update")
        self.centers = self._update_centers(X=X, n_clusters=self.n_clusters, m=self.m, weights=self.weights, _underflow=self._underflow)
        cur_center_list = np.argmax(self.centers, axis=0)
        if self.verbose: print("\tWeight matrix update")
        cur_weights = self.weights = self._update_weights(X=X, centers=self.centers, n_clusters=self.n_clusters, m=self.m, metric=self.metric, _underflow=self._underflow)
        cur_error = self._cal_error(X=X, weights=self.weights, centers=self.centers, n_clusters=self.n_clusters, metric=self.metric)

        n_iter = 1
        while n_iter < self.max_iter:
            if self.verbose: print("iter {n_iter}:".format(n_iter=n_iter))

            # 2-1. cluster update
            if self.verbose: print("\tCluster update")
            self.centers = self._update_centers(X=X, n_clusters=self.n_clusters, m=self.m, weights=self.weights, _underflow=self._underflow)
            new_center_list = np.argmax(self.centers, axis=0)
            # 2-2. weight matrix update
            if self.verbose: print("\tWeight matrix update")
            self.weights = self._update_weights(X=X, centers=self.centers, n_clusters=self.n_clusters, m=self.m, metric=self.metric, _underflow=self._underflow)
            new_weights = self.weights

            ## iter terminate condition check
            # if centroid doesn't change
            if "center" in optimization:
                if np.sum(~(cur_center_list==new_center_list)) < 1: 
                    print("centroid optimized")
                    break
                cur_center_list = new_center_list
            
            # if weights_change < min_criteria
            if "weights" in optimization:
                if np.linalg.norm(cur_weights - new_weights) <= self.min_weights_change:
                    print("weigths optimized")
                    break
                cur_weight = new_weights
            
            ## if errro_change < min_criteria
            if "error" in optimization:
                new_error = self._cal_error(X=X, weights=self.weights, centers=self.centers, n_clusters=self.n_clusters, metric=self.metric)
                if abs(cur_error - new_error) <= self.min_error_change: 
                    print("error optimized")
                    break
                cur_error = new_error            
            n_iter += 1
        
        if self.verbose: print("Clustering optimized in {n_iter} iteration".format(n_iter=n_iter))
        return self.weights
        
    
    def _init_weights_rand(self, n_points, n_clusters, _underflow):
        weights = np.zeros((n_points, n_clusters))
        while True:
            for i in range(n_points):
                rand_row = np.random.rand(n_clusters)
                # The sum of the weights of the clusters of a point(sum of row vector) must equal one.
                weights[i] = (np.exp(rand_row) + _underflow)/(np.sum(np.exp(rand_row)) + _underflow)

            # The sum of the weights of any cluster(sum of column vector) must not equal or exceed n_points.
            constraint_flag = True
            for j in range(n_clusters):
                if np.sum(weights[:, j])>=n_points: constraint_flag = False
            if constraint_flag: break
        return weights        
    
    def _update_centers(self, X, weights, n_clusters, m, _underflow):
        n_points = len(X)
        vector_size = len(X[0])
        centers = np.zeros((n_clusters, vector_size))

        weights_m = weights ** m
        centers = (np.dot(X.T, weights_m) / np.sum(weights_m, axis=0)).T
        return centers
    
    def _update_weights(self, X, centers, n_clusters, m, metric, _underflow):
        n_points = len(X)
        weights = np.zeros((n_points, n_clusters))

        dist_func = None
        if metric=="cosine": dist_func = self._cosine_distance
        elif metric=="mahalanobis": dist_func = self._mahalanobis_distance
        else: dist_func = self._euclidean_distance

        if m==1: update_exp = (1/_underflow)
        else: update_exp = (1/m-1)

        for i in range(n_points):
            w_i_nom = ((1 + _underflow)/((dist_func(centers, X[i])**2) + _underflow)) ** update_exp
            w_i_denom = np.sum(w_i_nom, axis=0)
            w_i = (w_i_nom + _underflow) / (w_i_denom + _underflow)
            weights[i] = w_i
        return weights
    
    def _cal_error(self, X, weights, centers, n_clusters, metric):
        dist_func = None
        if metric=="cosine": dist_func = self._cosine_distance
        elif metric=="mahalanobis": dist_func = self._mahalanobis_distance
        else: dist_func = self._euclidean_distance

        n_points = len(X)
        error = 0
        for i in range(n_points):
            for j in range(n_clusters):
                error += weights[i,j] * dist_func(X[i], centers[j])
        return error
    
    def _euclidean_distance(self, x, c):
        euclidean_distance = np.sqrt(np.sum((x - c)**2, axis=-1))
        return euclidean_distance
    
    def _cosine_distance(self, x, c):
        cosine_similarity = (np.dot(x, c) + self._underflow)/((np.linalg.norm(x)*np.linalg.norm(c)) + self._underflow)
        cosine_distance = 1 - cosine_similarity
        return cosine_distance
    
    def _mahalanobis_distance(self, x, c):
        cov_inverse = np.linalg.inv(np.cov(np.concatenate((self.X, self.centers)).T))
        x_bias = x - c
        mahalanobis_distance = np.dot(np.dot(x_bias, cov_inverse), x_bias.T)
        if len(x.shape) > 1 : mahalanobis_distance = mahalanobis_distance.diagonal()
        return mahalanobis_distance
