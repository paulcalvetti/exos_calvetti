import numpy as np
from filtering import *
from smoother import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from class_p import *

df = pd.read_csv(r'/Users/paulcalvetti/Documents/internship/exos/btcusd_15m/bctusd_15m_all.csv')
#df.set_index('time')
df['time'] = pd.to_datetime(df['time'])
df.sort_values(by = 'time', inplace=True)
df.set_index('time')

idx = (df.time >= '2015-01-01') & (df.time <= '2015-01-20')
#df2 = df[df['time'] > '2016-01-01' and df['time'] < '2017-01-01' ]
df2 = df[idx]





V = np.array([df2['log return'].tolist()])
plt.plot(df2.time,df2['vwap'])
plt.xticks(rotation=70)
plt.title("BTCUSD VWAP")
plt.show()
mean_val = np.mean(V)
print('mean_val',mean_val)
S = 2
T=np.size(V)


mu1h = np.empty(shape = [2,1],dtype = object)
mu1h[0] = np.array([[mean_val]])
mu1h[1] = np.array([[mean_val]])


sig0v = [.000005,.00002]
sig1h = np.empty(shape = [2,1,1],dtype = object)


pstgstm1ctm1 = np.empty(shape = [2,2,2])
pstgstm1ctm1[:,:,0] = np.array([[0.95,    .05],[0.05,    0.95]])
pstgstm1ctm1[:,:,1] = np.array([[1,    1.0000e-06],[1.0000e-06,    1]])



sig1h[0] = np.array([[.0000001]])
sig1h[1] = np.array([[.0000001]])




B0 = np.array([[1], [1]])
mu0v = [0, 0]



mu0h = np.empty(shape = [2,1],dtype = object)
mu0h[0] = np.array([[0]])
mu0h[1] = np.array([[0]])






sig0h = np.empty(shape = [2,1,1], dtype = object)

sig0h[0] =np.array([[.000001]])

sig0h[1] = np.array([[.000001]])



A = np.empty(shape = [2,1,1], dtype = object)

A[0] =  np.array([[1]])

A[1] = np.array([[1]])

B1 = np.empty(shape = [2,1],dtype = object)
B1[0] = np.array([[1]])
B1[1] = np.array([[1]])

mu1v = np.array([0,	0])
sig1v = np.array([.01,.01])










ps1 = np.array([[.6, .4]])


muh1 = np.empty(shape = [2,1],dtype = object)
muh1[0] = np.array([[0]])
muh1[1] = np.array([[0]])


sigh1 = np.empty(shape = [2,1,1])
sigh1[0] = np.array([[.00001]])

sigh1[1] = np.array([[.0000001]])






p = P(B0, mu0v,sig0v,mu0h,sig0h,A,B1,mu1v,sig1v,mu1h,sig1h,pstgstm1ctm1,ps1,muh1,sigh1)

f, F, w, alpha, loglik, reset_prob = filtering(p,V,7)




x,beta = RTSLinearSmoother(p,V,f,F,w,7)
print('x[5]',x[5])
mass_aprox = np.zeros(shape = [S,T])
for t in range(T):
    for s in range(S):
        mass_aprox[s, t] = sum(sum(x[t][s]))


fig, ax1 = plt.subplots()
ax1.plot(df2['time'],df2['vwap'])


ax1.set_title('BTCUSD VWAP')
ax2 = ax1.twinx()


plt.show()

dates = df2['time'].tolist()
plt.bar(range(np.size(mass_aprox[0])), mass_aprox[0],color='g', label='Low Variance', width=1,alpha=.5)
plt.bar(range(np.size(mass_aprox[0])),mass_aprox[1],color='r', bottom=mass_aprox[0],label = 'High Variance',width=1,alpha=.5)
plt.legend()
plt.twinx()
plt.plot(df2['vwap'].tolist(),label='VWAP')
plt.title("BTCUSD VWAP + Smoothed States")
plt.show()

