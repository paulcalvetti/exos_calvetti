import numpy as np
from filtering import *
from smoother import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r'/Users/paulcalvetti/Documents/internship/exos/eur_and_usd_fwdfwd_spread_trades.csv')[['Date_eur','EUR Spread']]
#df.set_index('Date_eur')
df['Date_eur'] = pd.to_datetime(df['Date_eur'])
df.sort_values(by = 'Date_eur', inplace=True)
df.set_index('Date_eur')
df['mvavg'] = df['EUR Spread'].rolling(50).mean()
df['abs_spread_less_mvavg'] = np.abs(df['EUR Spread'] - df['mvavg'])
df['tr_mr'] = np.log(1 + df['abs_spread_less_mvavg'].pct_change())

print('df[tr_mr]:',np.nanmin(df['tr_mr']))
idx = (df.Date_eur >= '2015-01-01') & (df.Date_eur <= '2017-01-01')



df2 = df[idx]


fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(df2['EUR Spread'])
axs[0].plot(df2['mvavg'],c='r')
axs[1].plot(df2['abs_spread_less_mvavg'])
plt.show()

V = np.array([df2['tr_mr'].tolist()])
#print(np.shape(V))
plt.plot(V[0])
plt.title("V: Input Data")
plt.show()
mean_val = np.mean(V)
print('mean_val',mean_val)


print(np.var(V[0]))
S = 2
T=np.size(V)


class P_testing():
    # p.B1, p.mu1v, p.sig1v, p.mu1h, p.sig1h, p.pstgstm1ctm1, p.ps1, p.muh1, p.sigh1
    def __init__(self, B0, mu0v, sig0v, mu0h, sig0h, A, B1, mu1v, sig1v, mu1h, sig1h, pstgstm1ctm1, ps1, muh1, sigh1):
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


def test_tr_mr(sig0v_val, abs_mu1h, sig1h_val):
    mu1h = np.empty(shape = [2,1],dtype = object)
    mu1h[0] = np.array([[abs_mu1h]])
    mu1h[1] = np.array([[-abs_mu1h]])

    mu0h = np.empty(shape = [2,1],dtype = object)
    mu0h[0] = np.array([[0]])
    mu0h[1] = np.array([[0]])
    sig0v = [sig0v_val,sig0v_val]


    sig1h = np.empty(shape = [2,1,1],dtype = object)
    sig1h[0] = np.array([[sig1h_val]])
    sig1h[1] = np.array([[sig1h_val]])










    B0 = np.array([[1], [1]])
    mu0v = [0, 0]










    sig0h = np.empty(shape = [2,1,1], dtype = object)

    sig0h[0] =np.array([[.01]])

    sig0h[1] = np.array([[.01]])



    A = np.empty(shape = [2,1,1], dtype = object)

    A[0] =  np.array([[1]])

    A[1] = np.array([[1]])

    B1 = np.empty(shape = [2,1],dtype = object)
    B1[0] = np.array([[1]])
    B1[1] = np.array([[1]])

    mu1v = np.array([0,	0])
    sig1v = np.array([.01,.01])







    pstgstm1ctm1 = np.empty(shape = [2,2,2])
    pstgstm1ctm1[:,:,0] = np.array([[0.95,    .05],[0.05,    0.95]])
    pstgstm1ctm1[:,:,1] = np.array([[1,    1.0000e-06],[1.0000e-06,    1]])


    ps1 = np.array([[.964, .036]])


    muh1 = np.empty(shape = [2,1],dtype = object)
    muh1[0] = np.array([[.5]])
    muh1[1] = np.array([[-.5]])


    sigh1 = np.empty(shape = [2,1,1])
    sigh1[0] = np.array([[.1]])

    sigh1[1] = np.array([[.1]])







    p = P_testing(B0, mu0v,sig0v,mu0h,sig0h,A,B1,mu1v,sig1v,mu1h,sig1h,pstgstm1ctm1,ps1,muh1,sigh1)

    f, F, w, alpha, loglik, reset_prob = filtering(p,V,8)



    s0 =[]
    s1=[]

    for t in range(T):
        s0.append(sum(w[t][0]))
        s1.append(sum(w[t][1]))

    s0=np.array(s0)
    s1=np.array(s1)


    #plt.bar(range(np.size(s0)), s0,color='g', label='Trending', width=1)
    #plt.bar(range(np.size(s0)),s1,color='r', bottom=s0,label = 'Mean Reverting', width=1)
    #plt.title('Swap Spread - 30 Day Moving Average')

    #plt.legend(loc = 'lower left')
    #plt.show()
    abs_diff = df2['abs_spread_less_mvavg'].tolist()
    logreturns = df2['tr_mr'].tolist()
    ret = []
    log_ret_trending =[]
    log_ret_mean_reverting = []
    for i in range(T - 1):
        if s0[i]>s1[i]: #greater probability of trending
            ret.append(abs_diff[i+1] - abs_diff[i])
            #log_ret_trending.append(logreturns[i+1])
            log_ret_trending.append(abs_diff[i+1] - abs_diff[i])

        else: # greater probability of mean reverting
            ret.append(abs_diff[i] - abs_diff[i + 1])
            #log_ret_mean_reverting.append(logreturns[i+1])
            log_ret_mean_reverting.append(abs_diff[i+1] - abs_diff[i])


    import seaborn as sns
    trending_mean = np.mean(log_ret_trending)
    mr_mean = np.mean(log_ret_mean_reverting)
    sns.distplot(log_ret_mean_reverting,hist=False,label = 'Mean Reverting', color='r')
    plt.axvline(mr_mean, color='r', linestyle='--', label='mean reverting mean')
    plt.title('Mean Reverting Return Dist')
    plt.axvline(trending_mean, color='g', linestyle='--', label='trending mean')



    sns.distplot(log_ret_trending,hist=False,label="Trending",color='g')
    plt.title('Trending vs MR Return Dist')
    plt.show()


    print('mr mean',np.mean(log_ret_mean_reverting))
    print('tr mean',np.mean(log_ret_trending))
    print('ret',np.sum(ret))
    print(len(abs_diff))
    print(len(s0))
    print(len(s1))
    return ret


test_tr_mr(.009, .005, .00001)
"""
test_tr_mr(.02, .01, .00001)
res_df = pd.DataFrame(columns=['sig0v_val', 'abs_mu1h', 'sumret', 'ret'])
return_dict = {}
x_vals = []
y_vals  = []
z_vals  = []
i=0
for sig0v_val in np.arange(.001,.021,.002):
    for abs_mu1h in np.arange(.005,.03,.005):
        a= test_tr_mr(sig0v_val, abs_mu1h, .00001)

        res_df.loc[i] = [sig0v_val,abs_mu1h,sum(a),a]
        i+=1


print(res_df)
print('testing')












prices = df2['EUR Spread'].tolist()
top_reset_prob = np.zeros(shape=np.shape(reset_prob))
topn = 50
sorted_indices = np.argsort(reset_prob[0])
print(sorted_indices)
sorted_indices_top = sorted_indices[-topn:]
max_val = max(prices)
for i in sorted_indices_top:
    top_reset_prob[0, i] = max_val
#plt.bar(range(np.size(np.sum(reset_prob.T, axis=1))), np.sum(top_reset_prob, axis=0), color='r')
#plt.plot(V[0])
spread = df2['EUR Spread'].tolist()
mvavg = df2['mvavg'].tolist()
#plt.plot(spread)
#plt.plot(mvavg,c='r')
#plt.title('50 Most Likely Resets')
#plt.show()

x,beta = RTSLinearSmoother(p,V,f,F,w,8)
print('x[5]',x[5])
mass_aprox = np.zeros(shape = [S,T])

for t in range(T):
    for s in range(S):
        print(sum(sum(x[t][s])))
        mass_aprox[s, t] = sum(sum(x[t][s]))


spread = df2['EUR Spread'].tolist()
mvavg = df2['mvavg'].tolist()
plt.bar(range(np.size(mass_aprox[0])),mass_aprox[0], width = 1, color = 'g')
plt.plot(spread)
plt.plot(mvavg,c='r')
plt.show()

abs_spr_mvavg = df2['abs_spread_less_mvavg'].tolist()
plt.bar(range(np.size(mass_aprox[0])),mass_aprox[0]/5, width = 1, color = 'g')
plt.plot(abs_spr_mvavg)
plt.show()

"""

