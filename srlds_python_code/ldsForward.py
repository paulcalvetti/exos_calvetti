import numpy as np
import functools

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
    muh = np.matmul(A,f)+meanH
    muv = np.matmul(B,muh)+meanV
    Shh = np.matmul(np.matmul(A,F),A.T)+CovH
    Svh = np.matmul(B,Shh)
    #Svv=B*Shh*B.T+CovV
    Svv=np.matmul(Svh,B.T)+CovV
    delta = v-muv

    #HOW TO REPLACE THE BACKSLASH-- UNIQUE SOLUTION???
    invSvvdel=delta/Svv

    fnew = muh+np.matmul(Svh.T,invSvvdel)
    #Fnew=Shh-Svh.T*(Svv\Svh) Fnew=0.5*(Fnew+Fnew.T)
    #K = Shh*B.T/Svv # Kalman Gain
    K = Svh.T/Svv # Kalman Gain
    tmp=np.identity(size(A))-np.matmul(K,B)
    Fnew =  functools.reduce(np.matmul, [tmp,Shh,tmp.T])+functools.reduce(np.matmul, [K,CovV,K.T])   # Joseph's form
    logpvgv = -0.5*np.matmul(delta.T,invSvvdel)-0.5*logdet(2*pi*Svv) ## THiS iS WHERE THE TiME iS SPENT.
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
    muh = np.matmul(A,f)+meanH
    Shtpt = np.matmul(A,F)
    #Shtptp=A*F*A.T+CovH
    Shtptp = np.matmul(Shtpt,A.T)+CovH

    ###BLACKSLASH
    leftA = (Shtpt.T)/Shtptp
    leftS = F - np.matmul(leftA,Shtpt)
    leftm = f - np.matmul(leftA,muh)
    gnew = np.matmul(leftA,g)+leftm
    leftAG = np.matmul(leftA,G)
    #Gnew = leftA*G*leftA.T+leftS Gnew=0.5*(Gnew+Gnew.T) # could also use Joseph's form if desired
    #Gpnew = leftA*G+gnew*g.T # smoothed <h_t h_{t+1}>
    Gnew = np.matmul(leftAG,leftA.T)+leftS
    Gnew= 0.5*(Gnew+Gnew.T) # could also use Joseph's form if desired
    Gpnew = leftAG+gnew*g.T # smoothed <h_t h_{t+1}>
    return gnew, Gnew, Gpnew


