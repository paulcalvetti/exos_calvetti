from data_creation import *
from utility_functions import *
from kalman_updates import *
from scipy.sparse import csc_matrix
import pdb


## FILTERING
#function [f,F,w,alpha,loglik]=filtering(p,V,numgaussians)
def filtering(p,V,numgaussians):


    ## constant
    ceq0=0
    ceq1=1


    ## initialise
    T = np.size(V)
    dh = np.size(p.mu0h[0])
    dv = np.size(p.mu0v[0])
    S = np.size(p.ps1)
    f = np.empty(T, dtype=object)
    F = np.empty(T, dtype=object)
    w = np.empty(T, dtype=object)
    alpha = np.empty(T, dtype=object)





    state_mean_est = np.zeros(shape = [S,T])
    prob_state = np.zeros(shape = [S,T])
    reset_prob = np.zeros(shape = [S,T])
    ft_break = np.zeros(shape = [S,T])








    for t in range(T):
        #changes min(t+1,numgaussians) to min(t+2,numgaussians) since we loop from 0:t-1 due to pythopn indexing
        f[t]=np.zeros(shape = [dh,S,min(t+2,numgaussians)])
        F[t] = np.zeros(shape = [dh,dh,S,min(t+2,numgaussians)])
        # W SHOULD BE SPARSE HOLDING IT AS EMPTY UNTIL I KNOW HOW TO CONVERT ********************
        #w[t] = csc_matrix(S,t+1,min(t+1,numgaussians))
        #w[t] = np.zeros(shape=[S, t + 2, min(t + 2, numgaussians)])
        w[t] = np.zeros(shape=[S, t+2])
        w[t][S-1,t+1] = min(t + 2, numgaussians)
        alpha[t] = np.zeros(shape = [S,1])





## first time-step (t=0)
    t=0

    #c1=1, reset
    ft = np.zeros(shape = [dh,S,2])
    Ft = np.zeros(shape = [dh,dh,S,2])
    logpvgsc = None
    logp = np.zeros(shape = [S,2])
    for s in range(S):
        ft_temp,Ft[:, :, s, 0], logpvgsc = forwardUpdate(np.zeros(shape = [dh,1]),np.zeros(shape = [dh,dh]),V[:,t],np.zeros(shape = [dh,dh]),p.B1[s],p.sig1h[s],p.sig1v[s],p.mu1h[s],p.mu1v[s])
        ft[:, s, 0] = ft_temp[:,0]
        logp[s,0] = np.log(p.ps1[0,s])+logpvgsc
        #c1=0, no reset
        ft_temp, Ft[:,:,s,1], logpvgsc = forwardUpdate(np.zeros(shape = [dh,1]), np.zeros(shape = [dh,dh]),V[:,t],np.zeros(shape = [dh,dh]),p.B0[s],p.sigh1[s],p.sig0v[s],p.muh1[s],p.mu0v[s])
        ft[:, s, 1] = ft_temp[:,0]

        logp[s,1] = np.log(p.ps1[0,s])+logpvgsc


    temp_w = condexp(np.reshape(logp, (1,np.product(logp.shape))))
    w[t][:]=np.reshape(temp_w, (np.size(temp_w) // 2, 2))

    # calculate likelihood
    loglik = logsumexp(np.reshape(logp, (np.size(logp), 1)), np.ones(shape = [1,2*S]))


    ls = np.empty(S, dtype = object)
    f = np.squeeze(f)
    for s in range(S):
        ls[s] = np.where(w[t][s,:] != 0)[0]
        f[t][:, s,:]= np.squeeze(ft[:, s, ls[s]])
        F[t][:,:, s,:] = np.squeeze(Ft[:,:, s, ls[s]])
        alpha[t][s] = np.sum(w[t][s,:])

    for t in range(1,T):
        #del ft, Ft
        ft = np.zeros(shape=[dh, S, min(t + 2, numgaussians+1)])
        Ft = np.zeros(shape=[dh, dh, S, min(t + 2, numgaussians+1)])

        del logp
        logp = - np.ones(shape = [S, t+2]) * np.inf


        for s in range(S):

            ft_temp, Ft[:, :, s, 0], logpvgscv = forwardUpdate(np.zeros(shape=[dh, 1]), np.zeros(shape=[dh, dh]), V[:, t], np.zeros(shape=[dh, dh]), p.B1[s], p.sig1h[s], p.sig1v[s], p.mu1h[s], p.mu1v[s])
            ft[:, s, 0] = ft_temp[:, 0]


            pstm1ctm1 = np.zeros(shape = [S-1, 2])





            if S == 1:
                logp[s, 0] = logpvgscv
            else:

                pstm1ctm1[:, ceq1] = w[t - 1][[i for i in range(S) if i != s], 0]

                pstm1ctm1[:, ceq0] = np.sum(w[t - 1][[i for i in range(S) if i != s], 1:])

                pstgstm1ctm1nots = np.reshape(p.pstgstm1ctm1[s, [i for i in range(S) if i != s], :], (S - 1, 2))
                logp[s, 0] = np.sum(np.log(np.dot(np.reshape(pstm1ctm1, (1,np.size(pstm1ctm1))),np.reshape(pstgstm1ctm1nots,(-1,1))))) + logpvgscv


            for j in range(len(ls[s])):

                ft_temp, Ft[:,:, s, j + 1], logpvgsciv = forwardUpdate(np.array([f[t-1][:,s,j]]).T,F[t-1][:,:,s,j],V[:,t],p.A[s],p.B0[s],p.sig0h[s],p.sig0v[s],p.mu0h[s],p.mu0v[s])



                ft[:, s, j + 1] = ft_temp[:, 0]


                i = ls[s][j]
                if i == 0: ctm1 = ceq1
                else: ctm1 = ceq0
                logp[s, i + 1] = sumlog(np.array([w[t-1][s,i], p.pstgstm1ctm1[s,s,ctm1]])) + logpvgsciv



        wt = np.reshape(condexp(np.reshape(logp, (1,np.size(logp)),order = 'F')),np.shape(logp),order = 'F')
        reset_prob[:,t] = wt[:, 0]
        for s in range(S):



            ls[s] = np.concatenate([np.array([0]),ls[s]+1])

            if t+1 >= numgaussians:
                idrop = np.argmin(wt[s,ls[s]])

                Z = np.sum(wt[s,ls[s]])


                ls[s] = np.array(np.delete(ls[s],idrop))
                w[t][s] = np.zeros(np.size(w[t][s]))
                for i in range(numgaussians):
                    #w[t] = np.zeros(shape=[S, t + 2])
                    w[t][s][ls[s][i]] = wt[s,ls[s][i]]
                w[t][s,:] = w[t][s,:]*Z/np.sum(w[t][s,:])

                f[t][:,s,:] = ft[:,s,:][:,[i for i in range(numgaussians+1) if i!= idrop]]

                F[t][:,:,s,:] = Ft[:,:,s,:][:,:,[i for i in range(numgaussians+1) if i!= idrop]]


            else:
                w[t] = wt
                f[t][:,s,:] = ft[:,s,ls[s]]
                F[t][:,:,s,:] = Ft[:,:,s,ls[s]]
            alpha[t][s] = np.sum(w[t][s,:])

        try:
            if S > 1:

                ft_no_break = np.squeeze(ft)
                ft_break[:,t] = ft_no_break[:,0]
                ft_no_break = ft_no_break[:,1:]
                w_save = np.reshape(w[t][np.nonzero(w[t])], np.shape(ft_no_break))
                state_mean_est[:,t] = np.sum(ft_no_break * w_save,axis = 1) / np.sum(w_save,axis = 1)
                prob_state[:,t] = np.sum(w_save,axis = 1)
            else:
                ft_no_break = np.squeeze(ft)
                ft_break[:,t] = ft_no_break[0]
                ft_no_break = ft_no_break[1:]
                w_save = np.reshape(w[t][np.nonzero(w[t])], np.shape(ft_no_break))
                state_mean_est[:,t] = np.sum(ft_no_break * w_save) / np.sum(w_save)
                prob_state[:,t] = np.sum(w_save)
        except:
            pass
        loglik += logsumexp(np.reshape(logp, (np.size(logp), 1)), np.ones(shape=[1, 2 * S]))
    reset_prob = sum(reset_prob,axis=1)
    return f, F, w, alpha, loglik, reset_prob
