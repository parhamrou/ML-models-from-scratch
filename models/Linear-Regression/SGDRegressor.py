import pandas as pd 
import numpy as np 

class SGDRegressor:
    """
    This is a SGD Regressor model which is implemented from scratch by me :) It consists of multiple methods for scaling, fitting, and training data.  
    """    
    
    def __init__(self, alpha):
        self.alpha = alpha
        self.error_hist = []
        self.coefs_ = None
        self.bias_ = None


    def fit(self, x: pd.DataFrame, y: pd.Series, iteration_num, error_hist_steps=100, standard_scale=True):
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
        
        self.gradient_descent(x, y, iteration_num, error_hist_steps)


    def predict(self, x:pd.DataFrame) -> np.array:
        x = x.values
        x = self.standard_scaling(x)
        m = x.shape[0]
        y_hats = np.zeros((m,))
        for i in range(m):
            y_hats[i] = np.dot(self.coefs_, x[i]) + self.bias_

        return y_hats


    def standard_scaling(self, x: np.array):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)

        return (x - mean) / std
    

    def squared_error(self, x: np.array, y: np.array):
        """
        This function is for calculating our erros in each step of our gradient descent algorithm. We use that to save 
        these values in a list that can be accessed later.
        """
        m = x.shape[0]
        error = 0.
        for i in range(m):
            y_hat = np.dot(self.coefs_, x[i]) + self.bias_
            error += (y_hat - y[i]) ** 2
        return error / (2 * m)


    def gradient_descent(self, x: np.array, y: np.array, iteration_num,  error_hist_steps):
        # We initialize our weights to zero
        n = x.shape[1]
        self.coefs_ = np.zeros((n,))
        self.bias_ = 0.


        for i in range(iteration_num):
            dj_dw, dj_db = self.gradient_function(x, y, self.coefs_, self.bias_)
            self.coefs_ -= self.alpha * dj_dw
            self.bias_ -= self.alpha * dj_db
            if (i + 1) % error_hist_steps == 0:
                error = self.squared_error(x, y)
                #print(f'Squared error after {i} iterations: {error}')
                self.error_hist.append(error)



    def gradient_function(self, x: np.array, y: np.array, w: np.array, b):
        """
        This method is for computing gradient function. It's used in our implementation of our gradient algorithm.
        """
        m, n = x.shape
        dj_dw = np.zeros((n,))
        dj_db = 0


        for i in range(m):
            # First we compute the error for this scatter regarding our current weights in w and b
            error = (np.dot(w, x[i]) + b) - y[i] 
            # Then we compute partial derivitives for our all w's 
            for j in range(n):
                dj_dw[j] += error * x[i, j]
            # We have to update b too
            dj_db += error 
        dj_dw /= m
        dj_db /= m

        return dj_dw, dj_db

    