from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
import numpy as np

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """
    bandwidth: 用于KDE模型的带宽
    kernel: 用于KDE模型的核函数名称
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.classes_ = None
        self.models_ = None
        self.logpriors_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        # 抓取標籤類別
        training_sets = [X[y == yi] for yi in self.classes_]
        # 訓練kde，計算機率密度
        self.models_ = [KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(Xi) for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0]) for Xi in training_sets]  # 計算先驗概率
        return self

    def predict_proba(self, X):
        logprobs = np.array([
            m.score_samples(X) for m in self.models_  # 計算輸入樣本作為每一個類在其所在位置出現的概率
        ]).T
        result = np.exp(logprobs + self.logpriors_)  # 概率的對數相加，等於概率相乘，乘每一個類的先驗概率
        np.seterr(invalid='ignore')
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]  # 找概率最大