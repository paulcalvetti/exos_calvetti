from utility_functions import *


class P():
     #p.B1, p.mu1v, p.sig1v, p.mu1h, p.sig1h, p.pstgstm1ctm1, p.ps1, p.muh1, p.sigh1
    def __init__(self, S, dh):



        self.B0 = np.empty(S, dtype=object)
        self.mu0v = np.empty(S, dtype=object)
        self.sig0v = np.empty(S, dtype=object)
        self.mu0h = np.empty(S, dtype=object)
        self.sig0h = np.empty(S, dtype=object)
        self.A = np.empty(S, dtype=object)
        self.B1 = np.empty(S, dtype=object)
        self.mu1v = np.empty(S, dtype=object)
        self.sig1v = np.empty(S, dtype=object)
        self.mu1h = np.empty(S, dtype=object)
        self.sig1h = np.empty(S, dtype=object)
        self.pstgstm1ctm1 = np.empty(S, dtype=object)
        self.ps1 = np.empty(S, dtype=object)
        self.muh1 = np.empty(S, dtype=object)
        self.sigh1 = np.empty(S, dtype=object)
        for s in range(S):
            self.B0[s] = np.random.normal(size = [1, dh])
            self.mu0v[s] = np.random.normal()
            self.sig0v[s] = 5
            self.mu0h[s] = np.random.normal(size = [dh, 1])
            self.sig0h[s] = randcov(dh)
            self.A[s] = np.random.normal(size=[dh,dh])
            self.B1[s] = np.random.normal(size=[1, dh])
            self.mu1v[s] = np.random.normal()
            self.sig1v[s] = 5
            self.mu1h[s] = np.random.normal(size=[dh, 1])
            self.sig1h[s] = randcov(dh)
            self.ps1[s] = np.random.rand()
            self.muh1[s] = np.random.normal(size=[dh,1])
            self.sigh1[s] = randcov(dh)



        #initialize self.pstgstm1ctm1
        for i in range(S):
            self.pstgstm1ctm1[i] = np.empty(S, dtype=object)
            for j in range(S):

                self.pstgstm1ctm1[i][j] = np.empty(2, dtype=object)
                self.pstgstm1ctm1[i][j][1] = .999999    #if the process just flipped, it doesn't flip again now
                self.pstgstm1ctm1[i][j][0] = np.random.rand()/2+.5   #if the process just flipped, it doesn't flip again now
        for i in range(S):

            for sp in range(S):
                if sp != s:
                    self.pstgstm1ctm1[sp][s] = np.empty(2,dtype=int)
                    self.pstgstm1ctm1[sp][s][1] = (1 - self.pstgstm1ctm1[s][s][1]) / (S - 1)
                    self.pstgstm1ctm1[sp][s][0] = (1 - self.pstgstm1ctm1[s][s][0]) / (S - 1)

        self.ps1 = self.ps1/np.sum(self.ps1)



p = P(4,5)
print(p.ps1[0])

print('end')



