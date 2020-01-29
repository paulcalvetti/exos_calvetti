import numpy as np
from filtering import *
from smoother import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv(r'/Users/paulcalvetti/Documents/internship/exos/btcusd_15m/bctusd_15m_all.csv')
#df.set_index('time')
df['time'] = pd.to_datetime(df['time'])
df.sort_values(by = 'time', inplace=True)
df.set_index('time')

idx = (df.time >= '2015-02-20') & (df.time <= '2015-03-01')
#df2 = df[df['time'] > '2016-01-01' and df['time'] < '2017-01-01' ]
df2 = df[idx]
print('here df2',df2)




V = np.array([df2['log return'].tolist()])
#print(np.shape(V))
plt.plot(df2.time,V[0])
plt.xticks(rotation=70)
plt.title("BTCUSD Log Returns")
plt.show()
mean_val = np.mean(V)
print('mean_val',mean_val)
S = 4
T=np.size(V)


mu1h = np.empty(shape = [S,1],dtype = object)
mu1h[0] = np.array([[mean_val]])
mu1h[1] = np.array([[mean_val]])
mu1h[2] = np.array([[mean_val]])
mu1h[3] = np.array([[mean_val]])

sig0v_low = .000005
sig0v_high = .0001
sig0v = [sig0v_low,sig0v_high,sig0v_low, sig0v_high]


sig1h_low = .000000001
sig1h_high = .00001

sig1h = np.empty(shape = [S,1,1],dtype = object)
sig1h[0] = np.array([[sig1h_low]])
sig1h[1] = np.array([[sig1h_low]])
sig1h[2] = np.array([[sig1h_high]])
sig1h[3] = np.array([[sig1h_high]])

sig0h_low = .000000001
sig0h_high = .00001

sig0h = np.empty(shape = [S,1,1], dtype = object)

sig0h[0] = np.array([[sig0h_low]])
sig0h[1] = np.array([[sig0h_low]])
sig0h[2] = np.array([[sig0h_high]])
sig0h[3] = np.array([[sig0h_high]])



pstgstm1ctm1 = np.empty(shape = [S,S,2])
pstgstm1ctm1[:,:,0] = np.array([[0.97,    .01,  .01, .01],[.01,.97,  .01, .01],[.01,.01,.97, .01],[.01,.01,.01,.97]])
pstgstm1ctm1[:,:,1] = np.array([[1,    1.0000e-06, 1.0000e-06, 1.0000e-06],[1.0000e-06,    1,  1.0000e-06,1.0000e-06],[1.0000e-06,  1.0000e-06,  1,1.0000e-06],[1.0000e-06,  1.0000e-06,  1.0000e-06,1]])








B0 = np.array([[1], [1], [1],[1]])
mu0v = [0, 0, 0,0]



mu0h = np.empty(shape = [S,1],dtype = object)
mu0h[0] = np.array([[0]])
mu0h[1] = np.array([[0]])
mu0h[2] = np.array([[0]])
mu0h[3] = np.array([[0]])










A = np.empty(shape = [S,1,1], dtype = object)

A[0] =  np.array([[1]])
A[1] = np.array([[1]])
A[2] = np.array([[1]])
A[3] = np.array([[1]])

B1 = np.empty(shape = [S,1],dtype = object)
B1[0] = np.array([[1]])
B1[1] = np.array([[1]])
B1[2] = np.array([[1]])
B1[3] = np.array([[1]])

mu1v = np.zeros(S)
sig1v = np.array([.00001,.00001,.00001,.00001])










ps1 = np.array([[1/S, 1/S, 1/S,1/S]])


muh1 = np.empty(shape = [S,1],dtype = object)
muh1[0] = np.array([[0]])
muh1[1] = np.array([[0]])
muh1[2] = np.array([[0]])
muh1[3] = np.array([[0]])

sigh1 = np.empty(shape = [S,1,1])
sigh1[0] = np.array([[.000001]])
sigh1[1] = np.array([[.0000001]])
sigh1[2] = np.array([[.0000001]])
sigh1[3] = np.array([[.0000001]])






class P_testing():
     #p.B1, p.mu1v, p.sig1v, p.mu1h, p.sig1h, p.pstgstm1ctm1, p.ps1, p.muh1, p.sigh1
    def __init__(self, B0, mu0v,sig0v,mu0h,sig0h,A,B1,mu1v,sig1v,mu1h,sig1h,pstgstm1ctm1,ps1,muh1,sigh1):
        self.B0 = B0
        self.mu0v = mu0v
        self.sig0v = sig0v
        self.mu0h = mu0h
        self.sig0h = sig0h
        self.A = A
        self.B1 = B1
        self.mu1v = mu1v
        self.sig1v = sig1v
        self.mu1h = mu1h
        self.sig1h = sig1h
        self.pstgstm1ctm1 = pstgstm1ctm1
        self.ps1 = ps1
        self.muh1 = muh1
        self.sigh1 = sigh1

p = P_testing(B0, mu0v,sig0v,mu0h,sig0h,A,B1,mu1v,sig1v,mu1h,sig1h,pstgstm1ctm1,ps1,muh1,sigh1)

f, F, w, alpha, loglik, reset_prob = filtering(p,V,5)

s0 =[]
s1=[]
s2=[]
s3=[]
for t in range(T):
    s0.append(sum(w[t][0]))
    s1.append(sum(w[t][1]))
    s2.append(sum(w[t][2]))
    s3.append(sum(w[t][3]))

"""
plt.plot(s0)
plt.title('S0')
plt.show()
plt.plot(s1)
plt.title('S1')

plt.show()
plt.plot(s2)
plt.title('S2')

plt.show()
"""
print('here',np.array(s0)+np.array(s1)+np.array(s2))
s0=np.array(s0)
s1=np.array(s1)

s2=np.array(s2)
s3=np.array(s3)

plt.bar(range(np.size(s0)), s0,color='g', label='Low Variance, Stable', width=1)
plt.bar(range(np.size(s0)),s1,color='r', bottom=s0,label = 'High Variance, Stable', width=1)
plt.bar(range(np.size(s0)),s2,color='b', bottom=s0+s1,label = 'Low Variance, Trending', width=1)
plt.bar(range(np.size(s0)),s3,color='c', bottom=s0+s1+s2,label = 'High Variance, Trending', width=1)


plt.legend(loc = 'lower left')
plt.show()

x,beta = RTSLinearSmoother(p,V,f,F,w,5)
print('x[5]',x[5])
mass_aprox = np.zeros(shape = [S,T])
for t in range(T):
    for s in range(S):
        mass_aprox[s, t] = sum(sum(x[t][s]))

print('mass check sum 1',np.sum(mass_aprox,axis=0))
fig, ax1 = plt.subplots()
ax1.plot(df2['time'],df2['vwap'],c='k')

# rotate and align the tick labels so they look better
#fig.autofmt_xdate()

# use a more precise date string for the x axis locations in the
# toolbar
#ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
ax1.set_title('BTCUSD VWAP')
ax2 = ax1.twinx()

plt.show()

plt.bar(range(np.size(mass_aprox[0])), mass_aprox[0],color='g', label='Low Variance, Stable', width=1)
plt.bar(range(np.size(mass_aprox[0])),mass_aprox[1],color='r', bottom=mass_aprox[0],label = 'High Variance, Stable', width=1)
plt.bar(range(np.size(mass_aprox[0])),mass_aprox[2],color='b', bottom=mass_aprox[0]+mass_aprox[1],label = 'Low Variance, Trending', width=1)
plt.bar(range(np.size(mass_aprox[0])),mass_aprox[3],color='c', bottom=mass_aprox[0]+mass_aprox[1]+mass_aprox[2],label = 'High Variance, Trending', width=1)

plt.legend(loc = 'lower left')
plt.show()


