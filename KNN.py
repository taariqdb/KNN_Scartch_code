from collections import Counter
import numpy as np
import math
def euclidean_distance(a,b):
    return np.sqrt(np.sum((a - b) ** 2))
    
class KNN:
    def __init__(self, K=3):
        self.K = K
    def fit(self, X,y):
        self.X_train = X
        self.y_train = y
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    def _predict(self, x):
        #computing the distance between x and all samples in X_train
        nearestpoint = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors 
        k_index = np.argsort(nearestpoint)[:self.K]
        # Extract the labels of the k nearest neighbor training samples
        kneighbor_labels = [self.y_train[i] for i in k_index]
        # return the most common class label
        most_frequent = Counter(kneighbor_labels).most_common(1)
        return most_frequent[0][0]
        