from builtins import range
from builtins import object
import numpy as np


class KNearestNeighbor(object):
    """a kNN classifier with L2 distance"""

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    # Assignment 1
    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sum((X[i] - self.X_train[j]) ** 2)
                # 작성하시오.
                # 각 이미지의 픽셀 차이를 sum()해서 각각을 행렬에 할당
        
        
        return dists


    # Assignment 2
    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # 작성하시오.
            # Numpy의 BroadCasting(가상 확장) 이용
            # 각 행(X[i])과 훈련 데이터와의 픽셀 차이 합을 구하기 위해 axis 인자 이용
            dists[i] = np.sum((X[i] - self.X_train) ** 2, axis=1)
        return dists

    # Assignment 3
    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # Note: (a - b)^2 = -2ab + a^2 + b^2
        # 작성하시오
        squared_X = np.square(X).sum(axis=1)
        squared_X_train = np.square(self.X_train).sum(axis=1)
        matrix_product = X @ self.X_train.T
        # 차원을 맞추기 위해 reshape 사용
        dists = squared_X.reshape(num_test, 1) + squared_X_train - 2 * matrix_product
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.y_train[dists[i].argsort()[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred
