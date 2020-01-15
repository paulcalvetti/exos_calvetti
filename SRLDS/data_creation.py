from utility_functions import *
from class_p import *
import matplotlib.pyplot as plt



#T, S, dh original parameters
def createData(T,S,dh):
    np.random.seed(1337)
    ceq0=1
    ceq1=2
    #initialize p using P class
    p = P(S, dh)

    #DIMENSIONS
    #not sure why this line needed so commented out for time being
    #dh=len(p.mu0h[0])  -based on class definition of p.mu0h this will always equal dh
    #dv = len(p.mu0v[0]) #not sure how this differs from saying dv = 0, maybe we will mutate p.mu0v
    dv = 1

    #INITIALISE
    s = np.zeros(T)
    H = np.zeros([dh,T])
    V = np.zeros([dv,T])


    t=0
    s[t] = int(randgen(p.ps1))
    s = np.int_(s) #convert np float64 to int64 to enable indexing
    H[:,t] = (p.mu1h[s[t]]+(np.linalg.cholesky(p.sig1h[s[t]])@np.random.normal(size = [dh, 1])))[:, 0]
    V[:,t] = p.B1[s[t]] @ H[:,t] + p.mu1v[s[t]]+np.linalg.cholesky(np.array([[p.sig0v[s[t]]]])) @ np.random.normal(size = [dv, 1])

    t = 1
    s[t] = randgen(p.pstgstm1ctm1[:][s[t-1]][1])
    s[t] = int(randgen(p.ps1))

    if s[t] == s[t - 1]: # states match --> no reset

        H[:, t] = (p.A[s[t]]@np.array([H[:,t-1]]).T + p.mu0h[s[t]] + np.linalg.cholesky(p.sig0h[s[t]]) @ np.random.normal(size = [dh,1]))[0]
        V[:,t] = p.B0[s[t]]@H[:,t] + p.mu0v[s[t]] + np.linalg.cholesky(np.array([[p.sig0v[s[t]]]])) @ np.random.normal(size = [dv,1])

    else: #states different --> reset
        H[:, t] = (p.mu1h[s[t]] + (np.linalg.cholesky(p.sig1h[s[t]]) @ np.random.normal(size=[dh, 1])))[:, 0]
        V[:, t] = p.B1[s[t]] @ H[:, t] + p.mu1v[s[t]] + np.linalg.cholesky(np.array([[p.sig0v[s[t]]]])) @ np.random.normal(size=[dv, 1])

    for t in range(2,T):

        if s[t-1] == s[t-2]:
            s[t] = randgen(p.pstgstm1ctm1[:][s[t - 1]][0])
        else:
            s[t] = randgen(p.pstgstm1ctm1[:] [s[t - 1]] [1])
        if s[t] == s[t - 1]:  # states match --> no reset

            H[:, t] = (p.A[s[t]] @ np.array([H[:, t - 1]]).T + p.mu0h[s[t]] + np.linalg.cholesky(
                p.sig0h[s[t]]) @ np.random.normal(size=[dh, 1]))[0]
            V[:, t] = p.B0[s[t]] @ H[:, t] + p.mu0v[s[t]] + np.linalg.cholesky(
                np.array([[p.sig0v[s[t]]]])) @ np.random.normal(size=[dv, 1])

        else:  # states different --> reset
            H[:, t] = (p.mu1h[s[t]] + (np.linalg.cholesky(p.sig1h[s[t]]) @ np.random.normal(size=[dh, 1])))[:, 0]
            V[:, t] = p.B1[s[t]] @ H[:, t] + p.mu1v[s[t]] + np.linalg.cholesky(np.array([[p.sig0v[s[t]]]])) @ np.random.normal(size=[dv, 1])

    return p,V,s,H


#p,V,s,H = createData(20,3,4)
#print(np.size(V))
#print(V)
#plt.plot(V[0])
#plt.show()



