from data_creation import *
from utility_functions import *
from kalman_updates import *
import copy



#function [x,beta]=RTSLinearSmoother(p,V,f,F,w,numgaussians)
def RTSLinearSmoother(p,V,f,F,w,numgaussians):
    ## constant
    ceq0=0
    ceq1=1

    ## initialise
    T = np.size(V)
    dh = np.size(p.mu0h[0])
    S = np.size(p.ps1)


    # t = T
    beta = np.empty(shape = [10],dtype = object)
    beta[T - 1] = np.sum(w[T - 1], axis = 1)


    ls = np.empty(S, dtype=object)
    x = np.empty(T, dtype=object)
    x[T - 1] = np.empty(shape = [S], dtype= object)
    meang = np.zeros(shape = [T,dh], dtype = object)
    gT = np.empty(S, dtype=object)
    GT = np.empty(S, dtype=object)
    jstp1 = np.ones(S)
    for s in range(S):
        ls[s] = np.where(w[T - 1][s,:] != 0)[0]
        x[T - 1][s] = w[T - 1][s,ls[s]]

        gT[s] = np.empty(len(ls[s]),dtype = object)
        GT[s] = np.empty(len(ls[s]),dtype = object)
        for i in range(len(ls[s])):
            gT[s][i] = f[T - 1][:, s, i]
            GT[s][i] = F[T - 1][:, :, s, i]
            meang[T - 1] = meang[T - 1] +x[T - 1][s][i]*gT[s][i]
    gtp1 = gT
    Gtp1 = GT
    istp1 = copy.copy(ls)


    ##t<T
    for t in reversed(range(T - 1)):

        xt = np.empty(S, dtype = object)
        Z = np.empty(S,dtype = object)
        Z2 = np.empty(S)
        for s in range(S):
            ls[s] = np.where(w[t][s,:] != 0)[0]
            xt[s] = np.zeros(shape = [len(ls[s]),S], dtype = object)
            for i in range(len(ls[s])):
                n_i_tp1 = i + 1 if (istp1[s][i]<=ls[s][i]) else i

                #copy in from t + 1

                if n_i_tp1 < np.size(istp1[s]) and ls[s][i] + 2 == istp1[s][n_i_tp1]:

                    xt[s][i,np.size(jstp1[s])] = x[t+1][s][n_i_tp1]

            #will likely be wrong when more than 2 states
            Z[s] = np.dot(np.delete(p.pstgstm1ctm1[s,:,ceq1],s), np.delete(w[t][:,0],s)) \
                   + np.sum(np.dot(np.delete(p.pstgstm1ctm1[s,:,ceq0],s),np.max(np.delete(w[t], s,axis = 0)[0,1:])))


            if istp1[s][0] == 0:

                Z2[s] = np.sum(x[t + 1][s])
            else:
                Z2[s] = 0


        print('Z: ',Z)
        print('Z2: ', Z2)
        for s in range(S):


            if ls[s][0] == 0:
                #need to modify when more than 2 states
                #to_mult_by = np.dot(np.divide(np.delete(p.pstgstm1ctm1[s,:,ceq1],s), np.delete(Z,s)),np.delete(Z2,s))
                xt[s][1,1] = np.dot(w[t][s,0] , np.dot(np.divide(np.delete(p.pstgstm1ctm1[s,:,ceq1],s), np.delete(Z,s)),np.delete(Z2,s)))
            isbiggerthan0 = np.where(ls[s] > 0)[0]


            #np.delete(p.pstgstm1ctm1[:, s, ceq0], s)
            #mult_by = ((p.pstgstm1ctm1[[:,s,ceq0]'./Z([1:s-1,s+1:S]))*Z2([1:s-1,s+1:S])')
            xt[s][isbiggerthan0, 1] = w[t][s,ls[s][isbiggerthan0]]

        return 0,1







#return x, beta


    