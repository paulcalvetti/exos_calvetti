import pandas as pd
import numpy as np
from data_creation import *
from filtering import *
from smoother import *
from class_p import *
#df = pd.read_csv(r'C:\Users\Documents\internship\exos\btc_15m_3days.csv')
df = pd.read_csv('btc_15m_3days.csv')
print(df)
V =np.array([df.mvavg_3h.to_list()])
V = 1000000*(V-V.mean())
V = V[0].tolist()

plt.plot(V)
plt.show()
T = np.size(V)
dh = 1
S = 2
p = P(2, 1)
V  = np.array([V])
print('V',V)

f, F, w, alpha, loglik = filtering(p,V,2)
w_1 = np.zeros(T)
w_2 = np.zeros(T)
for t in range(T):
    w_1[t],w_2[t] = np.sum(w[t], axis = 1)

print('w_1',w_1)
print('w_2',w_2)
print('w_1+w_2',w_1+w_2)

plt.plot(w_1,c='b')
plt.plot(w_2,c='r')
plt.show()
"""
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







import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

# Data
r = range(T)
raw_data = {'State 0': mass_aprox[0], 'State 1': mass_aprox[1]}
df = pd.DataFrame(raw_data)

# From raw value to percentage
totals = [1 for i, j in zip(df['State 0'], df['State 1'])]
greenBars = [i / j * 100 for i, j in zip(df['State 0'], totals)]
orangeBars = [i / j * 100 for i, j in zip(df['State 1'], totals)]


# plot
barWidth = 1
names = ('A', 'B', 'C', 'D', 'E')
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
# Create blue Bars


# Custom x axis
plt.xticks(r, names)
plt.xlabel("group")

# Show graphic
plt.show()


"""




