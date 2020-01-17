import pandas as pd
import numpy as np
from data_creation import *
from filtering import *
from smoother import *
#df = pd.read_csv(r'C:\Users\Documents\internship\exos\btc_15m_3days.csv')
df = pd.read_csv('btc_15m_3days.csv')
#print(df)
V =np.array([df.mvavg_3h.to_list()])
T = np.size(V)
dh = 5
S = 2
p,s,H = createData(V,T,S,dh)

f, F, w, alpha, loglik = filtering(p,V,2)

x,beta = RTSLinearSmoother(p,V,f,F,w,2)
print(x)
mass_aprox = np.zeros(shape = [S,T])
print('-----')
for t in range(T):
    for s in range(S):
        print(sum(sum(x[t][s])))
        #mass_aprox[s, t] =
        #mass_aprox[s,t] = sum(sum(x[t][s])) if sum(sum(x[t][s])) <=1 else 1
        mass_aprox[s, t] = sum(sum(x[t][s]))
print(mass_aprox)
print(np.sum(mass_aprox,axis = 0))


