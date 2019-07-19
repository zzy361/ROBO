

import numpy as np
from scipy import linalg
import scipy.optimize as sco


def blacklitterman(delta, weq, sigma, tau, P, Q, Omega):

    pi = weq.dot(sigma * delta)
    print('pi=',pi)

    ts = tau * sigma

    middle = linalg.inv(np.dot(np.dot(P,ts),P.T) + Omega)

    er = np.expand_dims(pi,axis=0).T + np.dot(np.dot(np.dot(ts,P.T),middle),(Q - np.expand_dims(np.dot(P,pi.T),axis=1)))
    print('er=',er)

    posteriorSigma = sigma + ts - ts.dot(P.T).dot(middle).dot(P).dot(ts)

    w = er.T.dot(linalg.inv(delta * posteriorSigma)).T

    lmbda = np.dot(linalg.pinv(P).T,(w.T * (1 + tau) - weq).T)
    return [er, w, lmbda]


def display(title,assets,res):
    er = res[0]
    w = res[1]
    lmbda = res[2]
    print('\n' + title)
    line = 'Country\t\t'
    for p in range(len(P)):
        line = line + 'P' + str(p) + '\t'
    line = line + 'mu\tw*'
    print(line)

    i = 0;
    for x in assets:
        line = '{0}\t'.format(x)
        for j in range(len(P.T[i])):
            line = line + '{0:.1f}\t'.format(100*P.T[i][j])

        line = line + '{0:.3f}\t{1:.3f}'.format(100*er[i][0],100*w[i][0])
        print(line)
        i = i + 1

    line = 'q\t\t'
    i = 0
    for q in Q:
        line = line + '{0:.2f}\t'.format(100*q[0])
        i = i + 1
    print(line)

    line = 'omega/tau\t'
    i = 0
    for o in Omega:
        line = line + '{0:.5f}\t'.format(o[i]/tau)
        i = i + 1
    print(line)

    line = 'lambda\t\t'
    i = 0
    for l in lmbda:
        line = line + '{0:.5f}\t'.format(l[0])
        i = i + 1
    print(line)


weq = np.array([0.016,0.022,0.052,0.055,0.116,0.124,0.615])

C = np.array([[ 1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
      [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
      [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
      [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
      [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
      [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
      [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]])

Sigma = np.array([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187])

refPi = np.array([0.039, 0.069, 0.084, 0.090, 0.043, 0.068, 0.076])
assets= {'Australia','Canada','France','Germany','Japan','UK','USA'}


V = np.multiply(np.outer(Sigma,Sigma), C)
print(V)


delta = 2.5


tau = 0.05
tauV = tau * V



