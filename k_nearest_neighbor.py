import numpy as np

class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """
        KNN은 학습 과정이 없고, 단순히 데이터를 저장만 합니다.
        """
        self.X_train = X
        self.y_train = y

    def compute_distances_two_loops(self, X):
        """
        두 개의 for-loop을 사용하여 모든 (test, train) 쌍의 L2 거리 계산
        Input: 
            X shape (num_test, D)
        Return:
            dists shape (num_test, num_train)
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                diff = X[i, :] - self.X_train[j, :]
                dists[i, j] = np.sqrt(np.sum(diff**2))
        return dists

    def compute_distances_one_loop(self, X):
        """
        하나의 for-loop만 사용하여 거리 계산
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            diff = self.X_train - X[i, :]   # (num_train, D)
            dists[i, :] = np.sqrt(np.sum(diff**2, axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        """
        완전 벡터화(no loop)로 거리 계산
        ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
        """
        # 각 벡터의 제곱합
        X_square = np.sum(X ** 2, axis=1).reshape(-1, 1)          # (num_test, 1)
        train_square = np.sum(self.X_train ** 2, axis=1).reshape(1, -1)  # (1, num_train)

        # 내적
        cross_term = X @ self.X_train.T                           # (num_test, num_train)

        # 거리 제곱 행렬
        dists_squared = X_square + train_square - 2 * cross_term

        # 수치적 안정성을 위해 음수값 보정
        dists_squared = np.maximum(dists_squared, 0)

        # 최종 거리 행렬
        return np.sqrt(dists_squared)


    def predict_labels(self, dists, k=1):
        """
        k개의 최근접 이웃을 찾아 다수결 투표로 label 예측
        Input:
            dists: 거리 행렬 (num_test, num_train)
            k: 최근접 이웃 개수
        Return:
            y_pred: 예측된 label 배열 (num_test,)
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, dtype=int)

        for i in range(num_test):
            # 거리가 가장 작은 k개 인덱스
            neighbors = np.argsort(dists[i])[:k]
            closest_y = self.y_train[neighbors]
            # 최빈값 (동점이면 가장 작은 label 선택)
            counts = np.bincount(closest_y)
            y_pred[i] = np.argmax(counts)
        return y_pred