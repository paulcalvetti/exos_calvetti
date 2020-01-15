import numpy as np

def logdet(A):
    #LOGDET Log determinant of a positive definite matrix computed in a numerically more stable manner
    u, s, v = np.linalg.svd(A)
    print(s+1.0e-20)
    l=np.sum(np.log(s+1.0e-20))
    return l

a=np.array([[.2,-.5],[1,-2]])
print(logdet(a))
