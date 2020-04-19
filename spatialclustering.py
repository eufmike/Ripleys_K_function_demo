#%%
import pickle
import scipy.stats
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

#%%
def PoissonPP(rt, Dx, Dy = None):
    '''
    Determines the number of events 'N' for a rectangular region,
    given the rate 'rt' and the dimensions, 'Dx', 'Dy'.
    Returns a <2xN> NumPy array.
    '''
    if Dy == None:
        Dy = Dx
    N = scipy.stats.poisson(rt*Dx*Dy).rvs()
    x = scipy.stats.uniform.rvs(loc = 0, scale = Dx, size = ((N, 1)))
    y = scipy.stats.uniform.rvs(loc = 0, scale = Dx, size = ((N, 1)))
    P = np.hstack((x, y))
    return(P)
    
rate, Dx = 0.2, 20
P = PoissonPP(rate, Dx).T
plt.figure(figsize=(10,10))
plt.scatter(P[0], P[1], edgecolor = 'b', facecolor='none', alpha =0.3)
plt.show()

#%%
def ThomasPP(kappa, sigma, mu, Dx):
    '''
    each forming a Poisson (mu) numbered cluster of points,
    having an isotropic Gaussian distribution with variance 'sigma'
    '''
    # Create a set of parent points form a Poisson(kappa)
    # distribution on the square region [0, Dx] * [0, Dx]
    parents = PoissonPP(kappa, Dx)
    # M is the number of parents
    M = parents.shape[0]
    # an empty list for the Thomas process points
    TP  = []
    # for each parent point
    for i in range(M):
        # determine a number of children accorfing to a Poisson(mu) distribution    
        N = scipy.stats.poisson(mu).rvs()
    # for each child point
        for j in range(N):
            # place a point centered on the location of the parent according 
            # to an isotropuc Gaussuan distribution with sigma variance
                pdf = scipy.stats.norm(loc = parents[i, :2], scale = (sigma, sigma))
                # add the child point to the list TP
                TP.append(list(pdf.rvs(2)))
    x, y = zip(*TP)
    pts = [x, y]
    return pts


aa = ThomasPP(kappa=3, sigma=0.2, mu=15, Dx=20)

#%%
plt.figure(figsize=(10,10))
plt.scatter(aa[0], aa[1], color='b', marker = '.', alpha = 0.2)
plt.show()

#%%
# Poisson test
import scipy.stats
import matplotlib.pyplot as plt

rt = 0.2
Dx = 20
Dy = 20
N = 80
print(N)
# N = scipy.stats.poisson(rt*Dx*Dy).rvs()
# print(N)
print(((N, 1)))

# 0, Dx means range, ((N, 1)) means dimentional structure
x = scipy.stats.uniform.rvs(loc = 0, scale = Dx, size = ((N, 1))) 
y = scipy.stats.uniform.rvs(loc = 0, scale = Dx, size = ((N, 1)))
P = np.hstack((x, y))
P_transpose = P.T

plt.scatter(P_transpose[0], P_transpose[1], edgecolor = 'b', facecolor='none', alpha =0.5)
plt.show()

print(P)
#%%
t_array = P
# print(t_array)

t = P[0, :2]
sigma = 0.2
N = 15
TP = []
for j in range(N):
            # place a point centered on the location of the parent according 
            # to an isotropuc Gaussuan distribution with sigma variance
                pdf = scipy.stats.norm(loc = t, scale = (sigma, sigma))
                # add the child point to the list TP
                TP.append(list(pdf.rvs(2)))

pts = list(zip(*TP))
print(pts)

#%%
mu = 10
N = scipy.stats.poisson(mu).rvs()
print(N)
