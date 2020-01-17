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


    #x = np.empty(T, dtype=object)
    #x[T - 1] = np.empty(shape = [S], dtype= object)
    x = np.zeros(shape = [T,S,S,S])


    meang = np.zeros(shape = [T,dh], dtype = object)
    #gT = np.empty(S, dtype=object)
    #GT = np.empty(S, dtype=object)
    gT = np.zeros(shape=[S, S, S, dh])
    GT = np.zeros(shape=[S, S, S, dh, dh])
    jstp1 = np.empty(S, dtype=object)

    for s in range(S):
        ls[s] = np.where(w[T - 1][s,:] != 0)[0]
        x[T - 1][s][:np.size(ls[s]),0] = w[T - 1][s,ls[s]]

        #gT[s] = np.empty(len(ls[s]),dtype = object)
        #GT[s] = np.empty(len(ls[s]),dtype = object)
        for i in range(len(ls[s])):
            gT[s,i,0] = f[T - 1][:, s, i]
            GT[s,i,0] = F[T - 1][:, :, s, i]
            meang[T - 1] = meang[T - 1] +x[T - 1][s,i,1]*gT[s][i][1]
        jstp1[s] = np.array([1])

    gtp1 = copy.deepcopy(gT)
    Gtp1 = copy.deepcopy(GT)
    istp1 = copy.deepcopy(ls)


    ##t<T
    for t in reversed(range(T - 1)):

        xt = np.zeros(S, dtype=object)

        Z = np.empty(S,dtype = object)
        Z2 = np.empty(S)

        for s in range(S):
            ls[s] = np.where(w[t][s,:] != 0)[0]
            xt[s] = np.zeros(shape = [len(ls[s]),np.size(jstp1[s]) + 1])

            for i in range(len(ls[s])):
                n_i_tp1 = i + 1 if (istp1[s][i]<=ls[s][i]) else i
                #copy in from t + 1
                if n_i_tp1 < np.size(istp1[s]) and ls[s][i] + 1 == istp1[s][n_i_tp1]:
                    print(xt[s][i,1:np.size(jstp1[s])])
                    print(x[t+1][s][1:n_i_tp1])

                    xt[s][i, 1:np.size(jstp1[s]) + 1] = x[t + 1][s,n_i_tp1,:np.size(jstp1[s])]
            #will likely be wrong when more than 2 states
                Z[s] = np.dot(np.delete(p.pstgstm1ctm1[s, :, ceq1], s), np.delete(w[t][:, 0], s)) \
                   + np.sum(np.dot(np.delete(p.pstgstm1ctm1[s, :, ceq0], s), np.sum(np.delete(w[t], s, axis=0)[0, 1:])))

            if istp1[s][0] == 0:

                Z2[s] = np.sum(x[t + 1][s][0])
            else:
                Z2[s] = 0

        for s in range(S):


            if ls[s][0] == 0:
                #need to modify when more than 2 states
                xt[s][0,0] = np.dot(w[t][s,0] , np.dot(np.divide(np.delete(p.pstgstm1ctm1[:,s,ceq1],s), np.delete(Z,s)),np.delete(Z2,s)))


            isbiggerthan0 = np.where(ls[s] > 0)[0]
            xt[s][isbiggerthan0, 0] = np.multiply(w[t][s,ls[s][isbiggerthan0]],np.divide(np.delete(p.pstgstm1ctm1[:,s,ceq0],s),np.delete(Z,s)))*np.delete(Z2,s)
        if numgaussians != np.inf: #may need to drop something
            for s in range(S):

                i,j = np.where(xt[s][:][:] !=0)
                v = np.zeros(len(i))
                for k in range(len(i)):
                    v[k] = xt[s][:][:][i[k],j[k]]
                indices = np.argsort(v)
                num = min(len(indices),numgaussians)
                indices =indices[-num:]
                xt[s].fill(0)
                for k in indices:
                    #assign xt and normalize to account for terms dropped
                    xt[s][i[k]][j[k]] = v[k] * sum(v) / sum(v[indices])




        x[t] = np.zeros(shape = [S, min(t+2,numgaussians), min(T - t + 2,numgaussians)])
        js = np.empty(S, dtype = object)
        for s in range(S):
            cols = np.where(np.sum(xt[s],axis=0) != 0)[0]
            x[t][s, :, :len(cols)] = xt[s][:,cols]
            #np.concatenate([np.array([0]),ls[s]+1])
            js[s] = np.concatenate([np.array([0]),jstp1[s]+1])
            js[s] = js[s][cols]

        beta[t] =np.sum(np.sum(x[t],axis = 2),axis=1)


        #shapes may need to be adjusted
        gt = np.zeros(shape = [S,S,S,dh])
        Gt = np.zeros(shape=[S, S,S, dh, dh])


        for s in range(S):
            for j in range(len(js[s])):
                jn = js[s][j]
                if jn == 0:
                    for i in range(len(ls[s])):

                        gt[s,i,0] = f[t][:,s,i]
                        Gt[s,i,0] = F[t][:,:,s,i]

                        meang[t] = meang[t] + np.dot(x[t][s,i,1],gt[s,i,0])
                else:
                    n_j_tp1 = 0
                    if np.size(jstp1[s]) > 1:
                        while jn - 1 != jstp1[s][n_j_tp1]:
                            n_j_tp1 += 1

                    for i in range(len(ls[s])):
                        if x[t][s, i, j]>0:
                            ln = ls[s][i]

                            n_i_tp1 = i + 1 if istp1[s][i] <= ln else i

                            #may need to modify first 2 terms (gtp1 and Gtp1) as dimension increase
                            #gt[s,i,j], Gt[s,i,j], ggpRGp = backwardUpdate(gtp1[s][n_i_tp1],Gtp1[s][n_i_tp1],f[t][:,s,i],F[t][:,:,s,i],p.A[s],p.sig0h[s], p.mu0h[s])
                            gt[s, i, j], Gt[s, i, j], ggpRGp = backwardUpdate(gtp1[s][n_i_tp1][n_j_tp1],Gtp1[s][n_i_tp1][n_j_tp1],f[t][:,s,i],F[t][:,:,s,i],p.A[s],p.sig0h[s], p.mu0h[s])
                            gt[0, 1, 0]
                            meang[t] = meang[t] + np.dot(x[t][s,i,j],gt[s][i,j])


        #gtp1 = copy.deepcopy(gt[:,:,0])
        #Gtp1 = copy.deepcopy(Gt[:,:,0])
        gtp1 = copy.deepcopy(gt)
        Gtp1 = copy.deepcopy(Gt)
        istp1 = copy.deepcopy(ls)
        jstp1 = copy.deepcopy(js)
    return x, beta





















#return x, beta


    