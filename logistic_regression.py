# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:45:25 2019

"""

import numpy as np

class LogisticRegression :
    step = 0
    n_iterations = 0
    
    def __init__ (self, step, n_iterations):
        self.step=step
        self.n_iterations=n_iterations
        
    def logistic(self,w,Xi):
        """ Compute the logistic function, given w and the features """
        a = np.dot(w.T,Xi)
        if(a >= 0 ):
            return 1/(1+np.exp(-a))
        else:
            return np.exp(a)/(1+np.exp(a))
    
    
    #x: traning data
    #y: traning resut
    #w: parameters
    #m: number of features
    #n: size of dataset
    
    def fit(self, x, y):

        n = x.shape[0]
        m = x.shape[1]
       
        #initialize m as ones 
        w = np.ones(m)
        #set step
        a = self.step
        count = 0 
        
        '''
        for i in range(m):
            x[:,i] = np.divide(x[:,i] - np.min(x[:,i]),np.max(x[:,i]) - np.min(x[:,i]))
        '''
        #doing gradient descent
        for _ in range(self.n_iterations):
            gradient = np.zeros((m))
            for i in range(n):
              gradient += (y[i] - self.logistic(w,x[i,:]))*x[i,:] 
            w = w + a * gradient
            count += 1 
        return w


    def predict(self, x, w):
        n=x.shape[0]        
        y= np.ones(n)
        for i in range(n):
            log_odds= np.dot(w.T,x[i,:])
            logistic_function= 1/(1+np.exp(-log_odds))
            if logistic_function >= 0.5:
                y[i] = 1
            else: 
                y[i] = 0
        return y
