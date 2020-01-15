import numpy as np

## UTILITY FUNCTIONS

def randcov(dh):
    x=0.001*np.random.normal(0,1,(dh,dh))
    y=np.triu(x) @ np.triu(x).T
    return y


####******************************************************************************
def randgen(p,m = 1,n = 1,s = None):
    # RANDGEN	Generates discrete random variables given the pdf
    #
    # Y = RANDGEN(P, M, N, S)
    #
    # Inputs :
    #	P : Relative likelihoods (or pdf) of symbols-- 1x(# states numpy matrix)
    #	M : Rows    <Default = 1>
    #	N : Columns <Default = 1>
    #	S : Symbols <Default 1:length(P)>

    # Change History :
    # Date		Time		Prog	Note
    # 06-Jul-1997	 4:36 PM	ATC	Created under MATLAB 5.0.0.4064

    # ATC = Ali Taylan Cemgil,
    # Bogazici University, Dept. of Computer Eng. 80815 Bebek Istanbul Turkey
    # e-mail : cemgil@boun.edu.tr
    if np.size(p) == 0: return np.array([[]])  # no samples if p is empty (DB)
    if s == None: s = np.array([range(np.size(p))])
    #if (nargin < 2) m = 1 end
    #if (nargin < 3) n = 1end
    #if (nargin < 4) s = 1:length(p)end

    c = np.cumsum(p)
    c_sum = np.sum(p)

    #if it a column matrix, not row need to flip c[0,-1] to c[-1,0]

    c = np.array([c[:].T / c_sum])
    N = m*n
    u = np.random.rand(N,1)
    # for i=1:N, y(i) = length(find(c<y(i)))+1 end
    #print(c)
    #print('--')
    #print(u)
    #print(np.where( c < np.matmul(u,np.ones([1,5]))))
    #print(np.where( c < u)[1])
    y = len(np.where( c < u)[1])

    #y = np.reshape(s[0,y-1], m, n)
    return y



#randgen sample inputs look:[0.984742716979373;0.00381432075515675;0.00381432075515675;0.00381432075515675;0.00381432075515675] look:[0.00616167127821327;0.975353314887147;0.00616167127821327;0.00616167127821327;0.00616167127821327]
#print(randgen(np.array([[0.9,0.9,0.9,0.9,0.5]]).T))






def logsumexp(a,b):
    #LOGSUMEXP Compute log(sum(exp(a).*b)) valid for large a
    # example: logsumexp([-1000 -1001 -998],[1 2 0.5])
    amax=np.max(a)
    A =  np.size(a)
    #test = np.exp(a-np.tile(amax, 1)) * b


    anew = amax + np.log(np.sum(np.exp(a-np.tile(amax, 1))))

    return anew


def logdet(A):
    #LOGDET Log determinant of a positive definite matrix computed in a numerically more stable manner
    #replace svd with just getting the value since input is always a number
    #u, s, v = np.linalg.svd(A)
    s=A
    l=np.sum(np.log(s+1.0e-20))
    return l

def condp(pin):
    #CONDP Make a conditional distribution from the matrix
    # pnew=condp(pin)
    #
    # Input : pin  -- a positive matrix pin
    # Output:  matrix pnew such that sum(pnew,1)=ones(1,size(p,2))

    p = pin/np.max(pin)

    ##TEMPORARY FOR TESTING
    eps=0

    p = p+eps # in case all unnormalised probabilities are zero
    pnew=p/np.tile(np.sum(p,axis=0),[np.shape(p)[0],1])
    return pnew

def condexp(logp):
    #CONDEXP  Compute p\propto exp(logp)
    pmax=np.max(logp)
    P =np.size(logp)

    pnew = condp(np.exp(logp.T-np.tile(pmax,[P,1])))

    return pnew


#print(condexp(np.array([[-1.85779181 -2.43236905 -6.28318398 -5.9973802 ]])))






def sumlog(x):
    #SUMLOG sum(log(x)) with a cutoff at 10e-200
    cutoff=10e-200
    x=x*(1+10e-200)
    x[x < cutoff] = cutoff

    lx=np.sum(np.log(x))
    return lx



