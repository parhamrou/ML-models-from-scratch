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


    def fit(self, x, y, iteration_number, error_hist_steps=100, standard_scale=True):
        """
        This method is used for fitting our data. 
        params:
        x: Our training features
        y: our target values in traning data
        iteration_num: Iteration steps for training our data
        error_hist_steps: After each n steps, the error is appended in a error history list
        standard_scaling: It performs standard scaling on our data
        """
        x = x.values
        y = y.values
        if standard_scale:
            x = self.standard_scaling(x)

        self.logistic_gradient_descent(x, y, iteration_number, error_hist_steps)



    def predict(self, x: pd.DataFrame):
        x = x.values
        x = self.standard_scaling(x)
        m = x.shape[0]
        predicts = np.zeros((m,))
        for i in range(m):
            predicts[i] = self.sigmoid(np.dot(x[i], self.coefs_) + self.bias_)

        return predicts
    


    def logistic_gradient_descent(self, x, y, iteration_num, error_hist_steps):
        n = x.shape[1]
        self.coefs_ = np.zeros((n,))
        self.bias_ = 0.

        for i in range(iteration_num):
            dj_dw, dj_db = self.logistic_gradient(x, y, self.coefs_, self.bias_)
            self.coefs_ -= self.alpha * dj_dw
            self.bias_ -= self.alpha * dj_db
            if (i + 1) %  error_hist_steps == 0:
                error = self.total_cost(x, y)
                self.cost_hist.append(error)


    def logistic_gradient(self, x: np.array, y: np.array, w: np.array, b):
        m, n = x.shape
        dj_dw = np.zeros((n,))
        dj_db = 0
        
        for i in range(m):
            # First we calculate the error for this row
            g = self.sigmoid(np.dot(self.coefs_, x[i]) + self.bias_)
            error = g - y[i]
            for j in range(n):
                dj_dw[j] += error * x[i, j]
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
        for i in range(m):
            z = self.sigmoid(np.dot(self.coefs_, x[i]) + self.bias_)
            cost += y[i] * (np.log(z)) + (1 - y[i]) * np.log(1 - z)
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
    

    def standard_scaling(self, x: np.array):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)

        return (x - mean) / std 