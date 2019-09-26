import numpy as np

def sinh(x):
    res = (1-np.exp(-2*x))/(2*np.exp(-x))
    return res

def cosh(x):
    res = (1+np.exp(-2*x))/(2*np.exp(-x))
    return res

class Benchmark_Agent:
    def __init__(self,k,tau,T,X):
        self.k = k  #k=sqrt(lambda_*sigma**2/eta)
        self.tau = tau
        self.T = T
        self.X = X

    def get_a_v(self,t): # t_j = t_j-0.5 = (j-0.5)*tau where j is the time step
        nj = (2*sinh(0.5*self.k*self.tau))/(self.X*sinh(self.k*self.T))*cosh(self.k*(self.T-t))
        return nj





