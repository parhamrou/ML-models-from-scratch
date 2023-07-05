import pandas as pd
import numpy as np

class LogisticRegression:
    """
    This class is an implementation for logistic regression algorithm from scratch using gradient descent and sigmoid function
    """
    def __init__(self, alpha):
        self.alpha = alpha
        self.cost_hist = []
        self.coefs_ = None
        self.bias_ = None

    def fit():
        pass


    def logistic_gradient_descent(x, y, iteration_num, error_hist_steps):
        


    def logistic_gradient(self, x: np.array, y: np.array, w: np.array, b):
        m, n = x.shape
        dj_dw = np.zeros((n,))
        dj_db = 0
        
        for i in range(m):
            # First we calculate the error for this row
            g = self.sigmoid(np.dot(self.coefs_, x) + self.bias_) - y[m]
            error = g - y[i]
            for j in range(n):
                dj_dw[n] += error * x[i, j]
            dj_db += error

        dj_dw /= m
        dj_db /= m
        
        return dj_dw, dj_db


    def total_cost(self, x: np.array, y: np.array):
        """
        This function is for evaluating the total error for our training set after each iteration. We use that for keeping the history 
        after each n iteration in our algorithm.
        parameters:
        x: The features array of our training set
        y: corresponding y's for our features
        """
        m = x.shape[0]
        cost = 0
        # We loop over all of the rows in training set and add their error to the total cost
        for m in range(m):
            z = np.dot(self.coefs_, x) + self.bias_
            cost += y * (np.log(-z)) + (1 - y) * (1 - np.log(-z))
        return cost / (-m)



    def sigmoid(self, z):
        """
        This function is the main part of our algorithm. It takes parameter z, and returns the returned value after applying.
        sigmoid function on that.
        parameters:
        z: scalar number
        """
        g = 1 / (1 + np.exp(-z))
        return g
