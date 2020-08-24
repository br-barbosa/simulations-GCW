
## Step 1 ##

import numpy as np
import matplotlib.pyplot as plt
import random as rd
import scipy.integrate as integrate
import os
os.makedirs('img')
np.random.seed(0)
rd.seed(0)

## Step 2 ##

L = 3.99
Lambda = np.linspace(-L, L, num=799)

b = 0.8   # beta
n = 25    # square's side
N = n*n   # number of vertices

g1 = lambda x: 4*x/np.pi
g2 = lambda x: 1/(x*np.pi + np.sin(2*np.pi*x))

G1 = integrate.quad(g1,0,0.5)[0]

def g(x):
  if abs(x)<=0.5:
    return integrate.quad(g1,0,x)[0]
  else:
    return G1 + integrate.quad(g2,0.5*np.sign(x),x)[0]


## Step 3 ##

def up_prob(S, past_val):
  q = np.zeros(len(Lambda))
  Z = 0

  for i, spin in enumerate(Lambda):
    q[i] = np.exp(N*b*g(S+(spin-past_val)/N) - spin**2)
    Z += q[i]
  return q/Z

def cdf(q, U, TV):
  F = np.add.accumulate(q)
  index = np.where(F >= U*TV)[0][0]-1
  return index

def create_img(X, Y, SX, SY, t, n, L):
  print(f"t={t}, d={round(np.sum(abs(X-Y)),3)}, Ham={np.sum(X != Y)}")

  fig, (gX, gY) = plt.subplots(ncols=2)

  gX.imshow(X.reshape(n,n), vmin=-L, vmax=L)
  gX.set_title(f'X : {round(SX[t],3)}')
  gX.axis('off')
  
  gY.imshow(Y.reshape(n,n), vmin=-L, vmax=L)
  gY.set_title(f'Y : {round(SY[t],3)}')
  gY.axis('off')
  
  plt.tight_layout()
  plt.savefig(f"img/C-{t}.png", bbox_inches='tight')
  plt.close()

## Step 4 ##

X = np.ones(N)
Y = np.random.randn(N)

SX = [np.average(X)]
SY = [np.average(Y)]

create_img(X, Y, SX, SY, t, n, L)

while ((t < N**2) & (abs(SX[t]-SY[t])>0.01)):
  v = np.random.randint(N)      
  SXtilde = SX[t] - X[v]/N
  SYtilde = SY[t] - Y[v]/N
  qXv = up_prob(SX[t], X[v])
  qYv = up_prob(SY[t], Y[v])
  TV = np.sum(np.absolute(qXv - qYv))/2

  if np.random.random() < TV:
    qX = qXv - qYv
    qY = qYv - qXv
    qX[qXv <= qYv] = 0
    qY[qYv <= qXv] = 0
    U = np.random.random()   # sample uniform
    X[v] = Lambda[cdf(qX, U, TV)]
    Y[v] = Lambda[cdf(qY, U, TV)]
  else:
    q = np.minimum(qXv,qYv)
    X[v] = Y[v] = rd.choices(population=Lambda, weights=q)[0]

  SX.append(SXtilde + X[v]/N)
  SY.append(SYtilde + Y[v]/N)
  t += 1

  if (t%(N)==0):
    create_img(X, Y, SX, SY, t, n, L)

create_img(X, Y, SX, SY, t, n, L)
