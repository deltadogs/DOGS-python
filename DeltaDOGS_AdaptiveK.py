#!/usr/bin/env python3
import numpy as np
from scipy.linalg import norm
import pandas as pd
import dogs
import scipy
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

'''
This script shows the Delta DOGS Adaptive K main code.

Contributors: Shahrouz Ryan Alimo, Muhan Zhao
Modified: Oct. 2017
'''

n = 2              # Dimenstion of input data
fun_arg = 2        # Type of function evaluation
plot_index = 0
iter_max = 50      # Maximum number of iterations based on each mesh
MeshSize = 1       # Represents the number of mesh refinement that algorithm will perform
num_iter = 0       # Represents how many iteration the algorithm goes
nff = 1            # Number of experiments
sc = "AdaptiveK"   # The type of continuous search function
# Calculate the Initial trinagulation points
Nm = 8             # Initial mesh grid size

# truth function
if fun_arg == 1:  # quadratic:
    fun = lambda x: 5 * norm(x - 0.3) ** 2
    y0 = 0.0  # targert value for objective function

elif fun_arg == 2:  # schewfel
    fun = lambda x: - sum(np.multiply(500 * x, np.sin(np.sqrt(abs(500 * x))))) / 250
    y0 = -1.6759 * n  # targert value for objective function
    xmin = 0.8419 * np.ones((1, n))

elif fun_arg == 3:  # rastinginn
    A = 3
    fun = lambda x: ( sum((x - 0.7) ** 2 - A * np.cos(2 * np.pi * (x - 0.7))) ) / 1.5

elif fun_arg == 5:  # schwefel + quadratic
    fun = lambda x: - x[0][0] / 2 * np.sin(np.abs(500*x[0][0])) + 10 * (x[1][0] - 0.92)**2
    lb = np.zeros(n)
    ub = np.ones(n)
    y0 = -0.44288
    xmin = np.array([0.89536, 0.94188])

elif fun_arg == 6: # Griewank function
    fun = lambda x: 1 + 1/4 * ((x[0][0]-0.67) ** 2 + (x[1][0] - 0.21) **2) - np.cos(x[0][0]) * np.cos(x[1][0]/np.sqrt(2))
    lb = np.zeros(n)
    ub = np.ones(n)
    y0 = 0.08026
    xmin = np.array([0.21875, 0.09375])

elif fun_arg == 7: # Shubert function
    tt = np.arange(1, 6)
    fun = lambda x: np.dot(tt, np.cos((tt + 1) * (x[0][0]-0.45) + tt)) * np.dot(tt, np.cos((tt + 1) * (x[1][0]-0.45) + tt))
    lb = np.zeros(n)
    ub = np.ones(n)
    y0 = -32.7533
    xmin = np.array([0.78125, 0.25])


xU = dogs.bounds(np.zeros([n, 1]), np.ones([n, 1]), n)
Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)

regret = np.zeros((nff, iter_max))
estimate = np.zeros((nff, iter_max))
datalength = np.zeros((nff, iter_max))
mesh = np.zeros((nff, iter_max))

for ff in range(nff):
    if fun_arg != 4:
        if n == 1:
            xE = np.array([[0.3, 0.5, 0.7]])
            xE = np.round(xE * Nm) / Nm  # quantize the points
        elif n == 2:
            xE = np.array([[0.875, 0.5, 0.25, 0.125, 0.75, 0.5], [0.25, 0.125, 0.875, 0.5, 0.875, 0.75]])
        else:
            xE = np.random.rand(n, n + 1)
            xE = np.round(xE * Nm) / Nm
        num_ini = xE.shape[1]
        yE = np.zeros(xE.shape[1])


        # Calculate the function at initial points
        for ii in range(xE.shape[1]):
            yE[ii] = fun(xE[:, ii].reshape(-1, 1))


    inter_par = dogs.Inter_par(method="NPS")
    yE_best = np.array([])


    for kk in range(MeshSize):

        for k in range(iter_max):

            num_iter += 1
            print('Total iter = ', num_iter, 'iteration k = ', k, ' kk = ', kk)

            K0 = np.ptp(yE, axis=0)  # scale the domain
            # y0 = scale_fun(y0,K0)

            [inter_par, yp] = dogs.interpolateparameterization(xE, yE, inter_par)


            ypmin = np.amin(yp)
            ind_min = np.argmin(yp)

            # Calcuate the unevaluated function:
            yu = np.zeros([1, xU.shape[1]])
            if xU.shape[1] != 0:
                for ii in range(xU.shape[1]):
                    tmp = dogs.interpolate_val(xU[:, ii], inter_par) - np.amin(yp)
                    yu[0, ii] = tmp / dogs.mindis(xU[:, ii], xE)[0]

            if xU.shape[1] != 0 and np.amin(yu) < 0:
                t = np.amin(yu)
                ind = np.argmin(yu)
                xc = xU[:, ind]
                yc = -np.inf
                xU = scipy.delete(xU, ind, 1)  # create empty array
            else:
                while 1:
                    # minimize s_c
                    xs, ind_min = dogs.add_sup(xE, xU, ind_min)
                    xc, yc = dogs.tringulation_search_bound(inter_par, xs, y0, K0, ind_min)
                    yc = yc[0, 0]
                    if dogs.interpolate_val(xc, inter_par) < min(yp):
                        xc = np.round(xc * Nm) / Nm
                        break
                    else:
                        xc = np.round(xc * Nm) / Nm
                        if dogs.mindis(xc, xE)[0] < 1e-6:
                            break
                        xc, xE, xU, success, _ = dogs.points_neighbers_find(xc, xE, xU, Bin, Ain)
                        if success == 1:
                            break
                        else:
                            yu = np.hstack([yu, (dogs.interpolate_val(xc, inter_par) - y0) / dogs.mindis(xc, xE)[0]])
                if xU.shape[1] != 0:
                    if dogs.mindis(xc, xE)[0] < 1e-6:
                        tmp = 1e+20
                    else:
                        tmp = (dogs.interpolate_val(xc, inter_par) - y0) / dogs.mindis(xc, xE)[0]
                        # both dicrete search function values need to be compared.
                    if (5*np.amin(yu)) < tmp and np.amin(yu)>0:
                        t = np.amin(yu)
                        ind = np.argmin(yu)
                        xc = xU[:, ind]
                        yc = -np.inf
                        xU = scipy.delete(xU, ind, 1)  # create empty array

                    # Minimize S_d ^ k(x)
            if dogs.mindis(xc, xE)[0] < 1e-6:
                Nm *= 2
                print('===============  MESH Refinement  ===================')

            else:
                print('xc = ', xc)
                print('yc = ', fun(xc))
                xE = np.hstack((xE, xc.reshape(-1, 1)))
                yE = np.hstack((yE, fun(xc.reshape(-1,1))))



            if 1:
                yE_best = np.hstack((yE_best, min(yE)))
                p_iter = np.zeros(num_iter)
                dogs.plot_delta_dogs(xE, yE, "Ddogs", sc, 0, fun_arg, p_iter, np.array([0, 1]), num_ini, Nm, 0, ff)
    if 1 and num_iter > 15:
        plt.figure()
        plt.title('Candidate point')
        plt.plot(np.arange(num_iter)+1, yE_best)
        plt.grid()
        plt.savefig('./figs/' + str(ff) + '/BestCandidate' +'.png', format='png', dpi=1000)
           #%%
