import numpy as np
import pandas as pd


class GaussianNB:

    def __init__(self) -> None:
        self.mu_ = None
        self.var_ = None
        self.epsilon_ = None


    def fit_predict(self, X: pd.DataFrame, y: pd.DataFrame):
        self.fit(X, y)
        return self.predict(X)


    def fit(self, X: pd.DataFrame, y=pd.DataFrame):
        """
        This method gets X which is an array of the features for each data point, and computes the mu and variance for each feature.
        """
        X = X.values
        y = y.values
        self.mu_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)
        prob_X = self.compute_probability(X)
        self.epsilon_ = self.compute_epsilon(prob_X, y)


    def predict(self, X: pd.DataFrame):
        X = X.values
        probs = self.compute_probability(X)
        return (probs < self.epsilon_)


    def compute_probability(self, X: np.array) -> np.array:
        features_probs = (1 / ((2 * np.pi) ** 0.5 * self.var_ ** 0.5)) * np.exp(-((X - self.mu_) ** 2) / 2 * self.var_ ** 2)
        result = np.prod(features_probs, axis=1)
        return result


    def compute_epsilon(self, X_probs: np.array, y: np.array) -> None:
        step_size = (np.max(X_probs) - np.min(X_probs)) / 1000
        best_epsilon = 0
        best_F1 = 0

        for epsilon in np.arange(np.min(X_probs), np.max(X_probs), step_size):
            predictions = (X_probs < epsilon)
            tp = np.sum((predictions == 1) & (y == 1))
            fn = np.sum((predictions == 0) & (y == 1))
            fp = np.sum((predictions == 1) & (y == 0))
            F1 = self.compute_F1_score(tp, fn, fp)
            if F1 > best_F1:   
                best_F1 = F1
                best_epsilon = epsilon
        return best_epsilon
    

    def compute_F1_score(self, tp, fn, fp):
        prec = self.compute_prec(tp, fp)
        rec = self.compute_rec(tp, fn)
        F1 = (2 * prec * rec) / (prec + rec)
        return F1


    def compute_prec(self, tp, fp):
        return tp / (tp + fp)


    def compute_rec(self, tp, fn):
        return tp / (tp + fn)