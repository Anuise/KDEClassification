from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
import numpy as np

class KDEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        """Bayesian generative classification based on KDE
        
        Parameters
        ----------
        bandwidth : float
            the kernel bandwidth within each class
        kernel : str
            the kernel name, passed to KernelDensity
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.classes_ = None
        self.models_ = None
        self.logpriors_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        # Get tag category
        # Output example: [0 1 2]
        training_sets = [X[y == yi] for yi in self.classes_]
        # Divide the data into N lists according to categories
        # training_sets shape: [self.classes_ size, x]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(Xi) for Xi in training_sets]
        # Use "sklearn KernelDensity" to calculate kernel density
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0]) for Xi in training_sets]
        # Calculate the prior probability of each class based on the sample size
        return self

    def predict_proba(self, X):
        logprobs = np.array([m.score_samples(X) for m in self.models_]).T
        # Calculate the probability of the input sample as each class appearing in its location
        result = np.exp(logprobs + self.logpriors_)
        # The logarithmic addition of the probabilities is equal to the multiplication of the probabilities.
        # Multiply the prior probability of each class.
        np.seterr(invalid='ignore')
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        # Find the largestest probability