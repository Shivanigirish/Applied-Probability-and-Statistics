import numpy as np

def generate_linear_data(n=100):
    x = np.random.randn(n,2)
    y = (x[:,0] + x[:,1] > 0).astype(int)
    return x,y

