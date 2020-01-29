import numpy as np
from filtering import *
from smoother import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from class_p import *

df = pd.read_csv(r'/Users/paulcalvetti/Documents/internship/exos/eur_and_usd_fwdfwd_spread_trades.csv')[['Date_eur','EUR Spread']]
#df.set_index('Date_eur')
df['Date_eur'] = pd.to_datetime(df['Date_eur'])
df.sort_values(by = 'Date_eur', inplace=True)
df.set_index('Date_eur')
df['spr_pct_change'] = df['EUR Spread'].pct_change()

idx = (df.Date_eur >= '2016-01-01') & (df.Date_eur <= '2018-01-01')



df2 = df[idx]



V = np.array([df2['spr_pct_change'].tolist()])




S = 1
T=np.size(V)


mu1h = np.empty(shape = [1,1],dtype = object)
mu1h[0] = np.array([[0]])
sig0v = [.001]
sig1h = np.empty(shape = [1,1,1],dtype = object)
sig1h[0] = np.array([[.000001]])




B0 = np.array([[1]])
mu0v = [0]

mu0h = np.empty(shape = [1,1],dtype = object)
mu0h[0] = np.array([[0]])
sig0h = np.empty(shape = [1,1,1], dtype = object)
sig0h[0] =np.array([[.0000001]])
A = np.empty(shape = [1,1,1], dtype = object)

A[0] =  np.array([[1]])



B1 = np.empty(shape = [1,1],dtype = object)
B1[0] = np.array([[1]])
mu1v = np.array([0])
sig1v = np.array([.000001])



pstgstm1ctm1 = np.empty(shape = [1,1,2])
pstgstm1ctm1[:,:,0] = np.array([[.99]])
pstgstm1ctm1[:,:,1] = np.array([[1]])




ps1 = np.array([[.01]])


muh1 = np.empty(shape = [1,1],dtype = object)
muh1[0] = np.array([[0]])



sigh1 = np.empty(shape = [1,1,1])
sigh1[0] = np.array([[.000001]])



p = P(B0, mu0v,sig0v,mu0h,sig0h,A,B1,mu1v,sig1v,mu1h,sig1h,pstgstm1ctm1,ps1,muh1,sigh1)

f, F, w, alpha, loglik, reset_prob = filtering(p,V,8)
print(np.arange(np.size(reset_prob)))
print(reset_prob)
plt.bar(np.arange(np.size(reset_prob)),reset_prob[0])
plt.plot(df2['EUR Spread'].tolist())
plt.show()
print('reset prib', reset_prob)




#x,beta = RTSLinearSmoother(p,V,f,F,w,8)
#print('x[5]',x[5])
#mass_aprox = np.zeros(shape = [S,T])
