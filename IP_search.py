"""
Created by Liu Zongying on June 2019

"""

import numpy as np
import mdp

def IP_searching_function(X, W_in, W_hat, eta, mu, sigma, nepochs):
    tol = 0.0000001
    # init IP
    ip_a = np.array(np.ones((len(W_in), 1)))
    ip_b = np.array(np.zeros((len(W_hat), 1)))

    for epoch in range(nepochs):
        old_a = ip_a
        old_b = ip_b

        for j in range(len(X)):
            if j == 0:
                xx = mdp.numx.dot(W_in, X[j, :])
            else:
                ss = mdp.numx.dot(W_hat,  last_state)
                sss = mdp.numx.dot(W_in, X[j, :])
                xx = addition(ss, sss)

            # save last echo states

            def multiply(a, b):
                qq = []
                for ii in range(len(a)):
                    qq.append(a[ii, :]*b[ii])
                return qq

            def addition(a, b):
                qqq = []
                for ii in range(len(a)):
                    qqq.append(a[ii, :]+b[ii])
                return qqq

            def devide(a, b):
                qq = []
                for ii in range(len(b)):
                    qq.append(a/b[ii, :])
                return qq

            def singlemultiply(a, b):
                qq = []
                for ii in range(len(b)):
                    qq.append(a*b[ii, :])
                return qq


            y = np.tanh(multiply(ip_a, xx)+ip_b)
            last_state = y

            zs = np.zeros((len(W_hat), 1))
            delta_b = -eta * (-(zs + mu/np.square(sigma)) + y/np.square(sigma)*(2*np.square(sigma) + 1 - multiply(y, y) + mu*y))
            ip_b = ip_b + delta_b
            delta_a = eta/ip_a + multiply(delta_b, xx)
            ip_a = ip_a + delta_a

        if np.linalg.norm(old_a-ip_a) < tol and np.linalg.norm(old_b - ip_b) < tol:
            break

    return ip_a, ip_b