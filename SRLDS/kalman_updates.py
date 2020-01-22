import numpy as np
import functools
from utility_functions import *
import numpy.linalg
## STANDARD KALMAN UPDATES
def forwardUpdate(f,F,v,A,B,CovH,CovV,meanH,meanV):
    #LDSFORWARDUPDATE Single Forward update for a Latent Linear Dynamical System (Kalman Filter)

    # [fnew Fnew logpvgv]=LDSforwardUpdate(f,F,v,A,B,CovH,CovV,meanH,meanV)
    #
    # inputs:
    # f : filterered mean p(h(t)|v(1:t))
    # F : filterered covariance p(h(t)|v(1:t))
    # v : observation v(t+1)
    # A : transition matrix
    # B : emission matrix
    # CovH : transition covariance
    # CovV : emission covariance
    # meanH : transition mean
    # meanV : emission mean
    #
    # Outputs:
    # fnew : : filterered mean p(h(t+1)|v(1:t+1))
    # Fnew : filterered covariance p(h(t+1)|v(1:t+1))
    # logpgvg : log p(v(t+1)|v(1:t))
    # print('A',A)
    # print('f',f)
    # print('A*f',np.dot(A,f))
    # print('np.array([meanH]).T',np.array([meanH])[0])


    muh = np.dot(A,f)+np.array([meanH]).T


    # print('muh',muh)
    muv = np.dot(B,muh) + meanV

    Shh = np.dot(np.dot(A,F),A.T)+CovH


    Svh = np.dot(B,Shh)

    #Svv=B*Shh*B.T+CovV

    Svv=np.dot(Svh,B.T)+CovV
    # print('v',v)
    # print('muv', muv)
    delta = v-muv

    #HOW TO REPLACE THE BACKSLASH-- UNIQUE SOLUTION???
    # print('del: ',(delta))
    # print('Svv:  ',(Svv))
    invSvvdel=delta/Svv

    fnew = (muh.T+Svh.T*invSvvdel).T
    #Fnew=Shh-Svh.T*(Svv\Svh) Fnew=0.5*(Fnew+Fnew.T)
    #K = Shh*B.T/Svv # Kalman Gain
    K = Svh.T/Svv # Kalman Gain
    tmp=np.identity(np.shape(A)[0])-np.dot(K,B)

    Fnew =  functools.reduce(np.dot, [tmp,Shh,tmp.T])+CovV*np.dot(K,K.T)   # Joseph's form


    logpvgv = -0.5*np.dot(delta.T,invSvvdel)-0.5*logdet(2*np.pi*Svv) ## THiS iS WHERE THE TiME iS SPENT.
    # print('orig code: ',logpvgv)
    # logpvgv = -0.5 * delta * invSvvdel - 0.5 * logdet(2 * np.pi * Svv)
    # print('here paul ',logpvgv)
    return fnew, Fnew, logpvgv



def backwardUpdate(g,G,f,F,A,CovH,meanH):
    #LDSBACKWARDUPDATE Single Backward update for a Latent Linear Dynamical System (RTS smoothing update)
    # [gnew Gnew Gpnew]=LDSbackwardUpdate(g,G,f,F,A,CovH,meanH)
    #
    # inputs:
    # g : smoothed mean p(h(t+1)|v(1:T))
    # G : smoothed covariance p(h(t+1)|v(1:T))
    # f : filterered mean p(h(t)|v(1:t))
    # F : filterered covariance p(h(t)|v(1:t))
    # A : transition matrix
    # CovH : transition covariance
    # CovV : emission covariance
    # meanH : transition mean
    #
    # Outputs:
    # gnew : smoothed mean p(h(t)|v(1:T))
    # Gnew : smoothed covariance p(h(t)|v(1:T))
    # Gpnew : smoothed cross moment  <h_t h_{t+1}|v(1:T)>
    muh = np.dot(A,f)+meanH
    Shtpt = np.dot(A,F)
    #Shtptp=A*F*A.T+CovH
    Shtptp = (np.dot(Shtpt,A.T)+CovH).astype('float64')
    print('Shtptp:',Shtptp)
    #leftA = np.dot(Shtpt.T,np.linalg.inv(Shtptp))
    leftA = np.dot(Shtpt.T,np.linalg.pinv(Shtptp))
    leftS = F - np.dot(leftA,Shtpt)
    leftm = f - np.dot(leftA,muh)
    gnew = np.dot(leftA,g)+leftm
    leftAG = np.dot(leftA,G)
    #Gnew = leftA*G*leftA.T+leftS Gnew=0.5*(Gnew+Gnew.T) # could also use Joseph's form if desired
    #Gpnew = leftA*G+gnew*g.T # smoothed <h_t h_{t+1}>
    Gnew = np.dot(leftAG,leftA.T)+leftS
    Gnew= 0.5*(Gnew+Gnew.T) # could also use Joseph's form if desired
    #Gpnew = leftAG+gnew*g.T # smoothed <h_t h_{t+1}> MY COMMENT
    Gpnew = leftAG + np.dot(np.array([gnew]).T, np.array([g]))
    return gnew, Gnew, Gpnew


