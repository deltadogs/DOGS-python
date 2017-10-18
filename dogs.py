import numpy as np
import scipy
import pandas as pd
from scipy import optimize
from scipy.spatial import Delaunay
import scipy.io as io
import os, inspect
import uq
from tr import transient_removal
import lorenz
import tr
import platform
from scipy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



'''MIT License
Copyright (c) 2017 Shahrouz Ryan Alimo
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. '''

'''
Collaborator: Muhan Zhao
Modified: Feb. 2017 '''


class pointsdata:
    index = 0
    cost_comp = 0
    simTime = 0
    N = 0
    xE = 0
    yE = 0
    sigma = 0
    yreal = 0


def bounds(bnd1, bnd2, n):
    #   find vertex of domain for a box domain.
    #   INPUT: n: dimension, bnd1: lower bound, bnd2: upper bound.
    #   OUTPUT: vertex of domain. 2^n number vector of n-D.
    #   Example:
    #           n = 3
    #           bnd1 = np.zeros((n, 1))
    #           bnd2 = np.ones((n, 1))
    #           bnds = bounds(bnd1,bnd2,n)
    #   Author: Shahoruz Alimohammadi
    #   Modified: Dec., 2016
    #   DELTADOGS package
    bnds = np.kron(np.ones((1, 2 ** n)), bnd2)
    for ii in range(n):
        tt = np.mod(np.arange(2 ** n) + 1, 2 ** (n - ii)) <= 2 ** (n - ii - 1) - 1
        bnds[ii, tt] = bnd1[ii];
    return bnds


def mindis(x, xi):
    '''
    calculates the minimum distance from all the existing points
    :param x: x the new point
    :param xi: xi all the previous points
    :return: [ymin ,xmin ,index]
    '''
    #
    # %
    # %
    # %
    x = x.reshape(-1, 1)
    y = float('inf')
    index = float('inf')
    x1 = np.copy(x) * 1e+20
    N = xi.shape[1]
    for i in range(N):
        y1 = np.linalg.norm(x[:, 0] - xi[:, i])
        if y1 < y:
            y = np.copy(y1)
            x1 = np.copy(xi[:, i])
            index = np.copy(i)
    return y, index, x1


def modichol(A, alpha, beta):
    #   Modified Cholesky decomposition code for making the Hessian matrix PSD.
    #   Author: Shahoruz Alimohammadi
    #   Modified: Jan., 2017
    n = A.shape[1]  # size of A
    L = np.identity(n)
    ####################
    D = np.zeros((n, 1))
    c = np.zeros((n, n))
    ######################
    D[0] = np.max(np.abs(A[0, 0]), alpha)
    c[:, 0] = A[:, 0]
    L[1:n, 0] = c[1:n, 0] / D[0]

    for j in range(1, n - 1):
        c[j, j] = A[j, j] - (np.dot((L[j, 0:j] ** 2).reshape(1, j), D[0:j]))[0, 0]
        for i in range(j + 1, n):
            c[i, j] = A[i, j] - (np.dot((L[i, 0:j] * L[j, 0:j]).reshape(1, j), D[0:j]))[0, 0]
        theta = np.max(c[j + 1:n, j])
        D[j] = np.array([(theta / beta) ** 2, np.abs(c[j, j]), alpha]).max()
        L[j + 1:n, j] = c[j + 1:n, j] / D[j]
    j = n - 1;
    c[j, j] = A[j, j] - (np.dot((L[j, 0:j] ** 2).reshape(1, j), D[0:j]))[0, 0]
    D[j] = np.max(np.abs(c[j, j]), alpha)
    return np.dot(np.dot(L, (np.diag(np.transpose(D)[0]))), L.T)


def circhyp(x, N):
    # circhyp     Circumhypersphere of simplex
    #   [xC, R2] = circhyp(x, N) calculates the coordinates of the circumcenter
    #   and the square of the radius of the N-dimensional hypersphere
    #   encircling the simplex defined by its N+1 vertices.
    #   Author: Shahoruz Alimohammadi
    #   Modified: Jan., 2017
    #   DOGS package

    test = np.sum(np.transpose(x) ** 2, axis=1)
    test = test[:, np.newaxis]
    m1 = np.concatenate((np.matrix((x.T ** 2).sum(axis=1)), x))
    M = np.concatenate((np.transpose(m1), np.matrix(np.ones((N + 1, 1)))), axis=1)
    a = np.linalg.det(M[:, 1:N + 2])
    c = (-1.0) ** (N + 1) * np.linalg.det(M[:, 0:N + 1])
    D = np.zeros((N, 1))
    for ii in range(N):
        M_tmp = np.copy(M)
        M_tmp = np.delete(M_tmp, ii + 1, 1)
        D[ii] = ((-1.0) ** (ii + 1)) * np.linalg.det(M_tmp)
        # print(np.linalg.det(M_tmp))
    # print(D)
    xC = -D / (2.0 * a)
    #	print(xC)
    R2 = (np.sum(D ** 2, axis=0) - 4 * a * c) / (4.0 * a ** 2)
    #	print(R2)
    return R2, xC


def data_merge(xi, yi):
    m = xi.shape[1]
    xnew = xi[:, 0].reshape(-1, 1)
    index = np.array([])
    for i in range(1, m):
        for j in range(xnew.shape[1]):
            if (xi[:, i] == xnew[:, j]).all():
                index = np.hstack((index, i))
                break
            if j == xnew.shape[1] - 1:
                xnew = np.hstack((xnew, xi[:, i].reshape(-1, 1)))
    ynew = np.delte(yi, index, 1)
    return xnew, ynew, index


########################################SURROGATE MODELS#################################################
class Inter_par():
    def __init__(self, method="NPS", w=0, v=0, xi=0, a=0):
        self.method = "NPS"
        self.w = []
        self.v = []
        self.xi = []
        self.a = []


def interpolateparameterization(xi, yi, inter_par):
    n = xi.shape[0]
    m = xi.shape[1]
    if inter_par.method == 'NPS':
        A = np.zeros((m, m))
        for ii in range(m):  # for ii = 0 to m-1 with step 1; range(1,N,1)
            for jj in range(m):
                A[ii, jj] = (np.dot(xi[:, ii] - xi[:, jj], xi[:, ii] - xi[:, jj])) ** (3.0 / 2.0)

        V = np.vstack((np.ones((1, m)), xi))
        A1 = np.hstack((A, np.transpose(V)))
        A2 = np.hstack((V, np.zeros((n + 1, n + 1))))
        yi = yi[np.newaxis, :]
        b = np.concatenate([np.transpose(yi), np.zeros((n + 1, 1))])
        A = np.vstack((A1, A2))
        wv = np.linalg.lstsq(A, b)
        wv = np.copy(wv[0])
        inter_par.w = wv[:m]
        inter_par.v = wv[m:]
        inter_par.xi = xi
        yp = np.zeros(m)
        for ii in range(m):
            yp[ii] = interpolate_val(xi[:, ii], inter_par)
        return inter_par, yp



def regressionparametarization(xi, yi, sigma, inter_par):
    # Notice xi, yi and sigma must be a two dimension matrix, even if you want it to be a vector.
    # or there will be error
    n = xi.shape[0]
    N = xi.shape[1]
    if inter_par.method == 'NPS':
        A = np.zeros((N, N))
        for ii in range(N):  # for ii =0 to m-1 with step 1; range(1,N,1)
            for jj in range(N):
                A[ii, jj] = (np.dot(xi[:, ii] - xi[:, jj], xi[:, ii] - xi[:, jj])) ** (3.0 / 2.0)
        V = np.concatenate((np.ones((1, N)), xi), axis=0)
        w1 = np.linalg.lstsq(np.dot(np.diag(1 / sigma), V.T), (yi / sigma).reshape(-1, 1))
        w1 = np.copy(w1[0])
        b = np.mean(np.divide(np.dot(V.T, w1) - yi.reshape(-1, 1), sigma.reshape(-1, 1)) ** 2)
        wv = np.zeros([N + n + 1])
        if b < 1:
            wv[N:] = np.copy(w1.T)
            rho = 1000
            wv = np.copy(wv.reshape(-1, 1))
        else:
            rho = 1.1
            fun = lambda rho: smoothing_polyharmonic(rho, A, V, sigma, yi, n, N)[0]
            rho = optimize.fsolve(fun, rho)
            b, db, wv = smoothing_polyharmonic(rho, A, V, sigma, yi, n, N)
        inter_par.w = wv[:N]
        inter_par.v = wv[N:N + n + 1]
        inter_par.xi = xi
        yp = np.zeros(N)
        while (1):
            for ii in range(N):
                yp[ii] = interpolate_val(xi[:, ii], inter_par)
            residual = np.max(np.divide(np.abs(yp - yi), sigma))
            if residual < 2:
                break
            rho *= 0.9
            b, db, wv = smoothing_polyharmonic(rho, A, V, sigma, yi, n, N)
            inter_par.w = wv[:N]
            inter_par.v = wv[N:N + n + 1]
    return inter_par, yp


def smoothing_polyharmonic(rho, A, V, sigma, yi, n, N):
    A01 = np.concatenate((A + rho * np.diag(sigma ** 2), np.transpose(V)), axis=1)
    A02 = np.concatenate((V, np.zeros(shape=(n + 1, n + 1))), axis=1)
    A1 = np.concatenate((A01, A02), axis=0)
    b1 = np.concatenate([yi.reshape(-1, 1), np.zeros(shape=(n + 1, 1))])
    wv = np.linalg.lstsq(A1, b1)
    wv = np.copy(wv[0])
    b = np.mean(np.multiply(wv[:N], sigma.reshape(-1, 1)) ** 2 * rho ** 2) - 1
    bdwv = np.concatenate([np.multiply(wv[:N], sigma.reshape(-1, 1) ** 2), np.zeros((n + 1, 1))])
    Dwv = np.linalg.lstsq(-A1, bdwv)
    Dwv = np.copy(Dwv[0])
    db = 2 * np.mean(np.multiply(wv[:N] ** 2 * rho + rho ** 2 * np.multiply(wv[:N], Dwv[:N]), sigma ** 2))
    return b, db, wv


def interpolate_hessian(x, inter_par):
    if inter_par.method == "NPS" or self.method == 1:
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        n = x.shape[0]
        N = xi.shape[1]
        g = np.zeros((n))
        n = x.shape[0]

        H = np.zeros((n, n))
        for ii in range(N):
            X = x[:, 0] - xi[:, ii]
            if np.linalg.norm(X) > 1e-5:
                H = H + 3 * w[ii] * ((X * X.T) / np.linalg.norm(X) + np.linalg.norm(X) * np.identity(n))
        return H


def interpolate_val(x, inter_par):
    # Each time after optimization, the result value x that optimization returns is one dimension vector,
    # but in our interpolate_val function, we need it to be a two dimension matrix.
    x = x.reshape(-1, 1)
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        x1 = np.copy(x)
        x = pd.DataFrame(x1).values
        S = xi - x
        return np.dot(v.T, np.concatenate([np.ones((1, 1)), x], axis=0)) + np.dot(w.T, (
            np.sqrt(np.diag(np.dot(S.T, S))) ** 3))


def interpolate_grad(x, inter_par):
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        n = x.shape[0]
        N = xi.shape[1]
        g = np.zeros([n, 1])
        x1 = np.copy(x)
        x = pd.DataFrame(x1).values
        for ii in range(N):
            X = x - xi[:, ii].reshape(-1, 1)
            g = g + 3 * w[ii] * X * np.linalg.norm(X)
        g = g + v[1:]

    return g


def inter_cost(x,inter_par):
    x = x.reshape(-1, 1)
    M = interpolate_val(x, inter_par)
    DM = interpolate_grad(x, inter_par)
    return M, DM.T


#TODO inter_min has been fixed
def inter_min(x, inter_par, lb=[], ub=[]):   # TODO fixed for now, quite different with matlab !!!
    # Find the minimizer of the interpolating function starting with x
    rho = 0.9  # backtracking parameter  # TODO didnt use rho parameter
    n = x.shape[0]
    x0 = np.zeros((n, 1))
    x = x.reshape(-1, 1)
    objfun = lambda x: interpolate_val(x, inter_par)
    grad_objfun = lambda x: interpolate_grad(x, inter_par)
    opt = {'disp': False}
    bnds = tuple([(0, 1) for i in range(int(n))])
    res = optimize.minimize(objfun, x0, jac=grad_objfun, method='TNC', bounds=bnds, options=opt)
    x = res.x
    y = res.fun
    return x, y


#################################### Constant K method ####################################
def tringulation_search_bound_constantK(inter_par, xi, K, ind_min):
    '''
    This function is the core of constant-K continuous search function.
    :param inter_par: Contains interpolation information w, v.
    :param xi: The union of xE(Evaluated points) and xU(Support points)
    :param K: Tuning parameter for constant-K, K = K*K0. K0 is the range of yE.
    :param ind_min: The correspoding index of minimum of yE.
    :return: The minimizer, xc, and minimum, yc, of continuous search function.
    '''
    inf = 1e+20
    n = xi.shape[0]
    # Delaunay Triangulation
    if n == 1:
        sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
        tri = np.zeros((xi.shape[1] - 1, 2))
        tri[:, 0] = sx[:xi.shape[1] - 1]
        tri[:, 1] = sx[1:]
        tri = tri.astype(np.int32)
    else:
        options = 'Qt Qbb Qc' if n <= 3 else 'Qt Qbb Qc Qx'
        tri = scipy.spatial.Delaunay(xi.T, qhull_options=options).simplices
        keep = np.ones(len(tri), dtype=bool)
        for i, t in enumerate(tri):
            if abs(np.linalg.det(np.hstack((xi.T[t], np.ones([1, n + 1]).T)))) < 1E-15:
                keep[i] = False  # Point is coplanar, we don't want to keep it
        tri = tri[keep]
    # Search the minimum of the synthetic quadratic model
    Sc = np.zeros([np.shape(tri)[0]])
    Scl = np.zeros([np.shape(tri)[0]])
    for ii in range(np.shape(tri)[0]):
        # R2-circumradius, xc-circumcircle center
        R2, xc = circhyp(xi[:, tri[ii, :]], n)
        # x is the center of the current simplex
        x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
        Sc[ii] = interpolate_val(x, inter_par) - K * (R2 - np.linalg.norm(x - xc) ** 2)
        if np.sum(ind_min == tri[ii, :]):
            Scl[ii] = np.copy(Sc[ii])
        else:
            Scl[ii] = inf
    # Global one
    t = np.min(Sc)
    ind = np.argmin(Sc)
    R2, xc = circhyp(xi[:, tri[ind, :]], n)
    x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))
    xm, ym = Constant_K_Search(x, inter_par, xc, R2, K)
    # Local one
    t = np.min(Scl)
    ind = np.argmin(Scl)
    R2, xc = circhyp(xi[:, tri[ind, :]], n)
    # Notice!! ind_min may have a problem as an index
    x = np.copy(xi[:, ind_min])
    xml, yml = Constant_K_Search(x, inter_par, xc, R2, K)
    if yml < ym:
        xm = np.copy(xml)
        ym = np.copy(yml)
    return xm, ym


def Constant_K_Search(x0, inter_par, xc, R2, K, lb=[], ub=[]):
    n = x0.shape[0]
    costfun = lambda x: Contious_search_cost(x, inter_par, xc, R2, K)[0]
    costjac = lambda x: Contious_search_cost(x, inter_par, xc, R2, K)[1]
    opt = {'disp': False}
    bnds = tuple([(0, 1) for i in range(int(n))])
    res = optimize.minimize(costfun, x0, jac=costjac, method='TNC', bounds=bnds, options=opt)
    x = res.x
    y = res.fun
    return x, y


# Value of constant K search
def Contious_search_cost(x, inter_par, xc, R2, K):
    x = x.reshape(-1, 1)
    M = interpolate_val(x, inter_par) - K * (R2 - np.linalg.norm(x - xc) ** 2)
    DM = interpolate_grad(x, inter_par) + 2 * K * (x - xc)
    return M, DM.T


#################################### Adaptive K method ####################################
def tringulation_search_bound(inter_par, xi, y0, K0, ind_min):
    inf = 1e+20
    n = xi.shape[0]
    xm, ym = inter_min(xi[:, ind_min], inter_par)
    ym = ym[0, 0]  # If using scipy package, ym would first be a two dimensions array.
    sc_min = inf
    # cse=1
    if ym > y0:
        ym = inf
    # cse =2
    # construct Deluanay tringulation
    if n == 1:
        sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
        tri = np.zeros((xi.shape[1] - 1, 2))
        tri[:, 0] = sx[:xi.shape[1] - 1]
        tri[:, 1] = sx[1:]
        tri = tri.astype(np.int32)
    else:
        options = 'Qt Qbb Qc' if n <= 3 else 'Qt Qbb Qc Qx'
        tri = scipy.spatial.Delaunay(xi.T, qhull_options=options).simplices
        keep = np.ones(len(tri), dtype=bool)
        for i, t in enumerate(tri):
            if abs(np.linalg.det(np.hstack((xi.T[t], np.ones([1, n + 1]).T)))) < 1E-15:
                keep[i] = False  # Point is coplanar, we don't want to keep it
        tri = tri[keep]

    Sc = np.zeros([np.shape(tri)[0]])
    Scl = np.zeros([np.shape(tri)[0]])
    for ii in range(np.shape(tri)[0]):
        R2, xc = circhyp(xi[:, tri[ii, :]], n)
        # if R2 != np.inf:
        if R2 < inf:
            # initialze with body center of each simplex
            x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
            Sc[ii] = (interpolate_val(x, inter_par) - y0) / (R2 - np.linalg.norm(x - xc) ** 2)
            if np.sum(ind_min == tri[ii, :]):
                Scl[ii] = np.copy(Sc[ii])
            else:
                Scl[ii] = inf
        else:
            Scl[ii] = inf
            Sc[ii] = inf

    # Global one
    ind = np.argmin(Sc)
    R2, xc = circhyp(xi[:, tri[ind, :]], n)
    x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))
    xm, ym = Adoptive_K_Search(x, inter_par, xc, R2, y0, K0)
    # Local one
    ind = np.argmin(Scl)
    R2, xc = circhyp(xi[:, tri[ind, :]], n)
    # Notice!! ind_min may have a problen as an index
    x = np.copy(xi[:, ind_min])
    xml, yml = Adoptive_K_Search(x, inter_par, xc, R2, y0, K0)
    if yml < 2 * ym:
        xm = np.copy(xml)
        ym = np.copy(yml)
    return xm, ym


def Adoptive_K_Search(x0, inter_par, xc, R2, y0, K0, lb=[], ub=[]):
    # Find the minimizer of the search fucntion in a simplex
    n = x0.shape[0]
    costfun = lambda x: AdaptiveK_search_cost(x, inter_par, xc, R2, y0, K0)[0]
    costjac = lambda x: AdaptiveK_search_cost(x, inter_par, xc, R2, y0, K0)[1]
    opt = {'disp': False}
    bnds = tuple([(0, 1) for i in range(int(n))])
    # res = optimize.minimize(costfun, x0, jac=costjac, method='TNC', bounds=bnds, options=opt)
    res = optimize.minimize(costfun, x0, jac=costjac, method='TNC', bounds=bnds, options=opt)
    x = res.x
    y = res.fun
    return x, y


def AdaptiveK_search_cost(x, inter_par, xc, R2, y0, K0):
    x = x.reshape(-1, 1)
    p = interpolate_val(x, inter_par)
    e = R2 - np.linalg.norm(x - xc) ** 2
    # Search function value
    M = - e * K0 / (p - y0)
#    M = (p - y0) / e

    # Gradient of search
    gp = interpolate_grad(x, inter_par)
    # ge = - 2 * np.linalg.norm(x - xc)
    ge = - 2 * (x - xc)
    DM = - ge * K0 / (p - y0) + K0 * e * gp / (p - y0) ** 2
    # DM = gp / e - ge * (p - y0) / e ** 2
#    if p < y0:
#        M = -M * np.inf
#        DM = gp * 0
    return M, DM.T


############################### Cartesian Lattice functions ######################
def add_sup(xE, xU, ind_min):
    '''
    To avoid duplicate values in support points for Delaunay Triangulation.
    :param xE: Evaluated points.
    :param xU: Support points.
    :param ind_min: The minimum point's index in xE.
    return: Combination of xE and xU, and the unique elements without changing orders.
    '''
    xmin = xE[:, ind_min]
    xs = np.hstack((xE, xU))
    n = xs.shape[1]
    # Construct the concatenate of xE and xU and return the array that every column is unique
    x_uni = xs[:, 0].reshape(-1, 1)
    for i in range(1, n):
        for j in range(x_uni.shape[1]):
            if ( xs[:, i] == x_uni[:, j] ).all():
                break
            if j == x_uni.shape[1] - 1:
                x_uni = np.hstack(( x_uni, xs[:, i].reshape(-1, 1) ))
    # Find the minimum point's index: ind_min
    for i in range(x_uni.shape[1]):
        if (x_uni[:, i] == xmin).all():
            ind_min_new = i
            break
    return x_uni, ind_min_new


def ismember(A, B):
    return [np.sum(a == B) for a in A]


def points_neighbers_find(x, xE, xU, Bin, Ain):
    '''
    This function aims for checking whether it's activated iteration or inactivated.
    If activated: perform function evaluation.
    Else: add the point to support points.
    :param x: Minimizer of continuous search function.
    :param xE: Evaluated points.
    :param xU: Support points.
    :return: x, xE is unchanged.
                If success == 1: evaluate x.
                Else: Add x to xU.
    '''
    x = x.reshape(-1, 1)
    x1 = mindis(x, np.concatenate((xE, xU), axis=1))[2].reshape(-1, 1)
    active_cons = []
    b = Bin - np.dot(Ain, x)
    for i in range(len(b)):
        if b[i][0] < 1e-3:
            active_cons.append(i + 1)
    active_cons = np.array(active_cons)

    active_cons1 = []
    b = Bin - np.dot(Ain, x1)
    for i in range(len(b)):
        if b[i][0] < 1e-3:
            active_cons1.append(i + 1)
    active_cons1 = np.array(active_cons1)
    # Explain the following two criterias:
    # The first means that x is an interior point.
    # The second means that x and x1 have exactly the same constraints.
    if len(active_cons) == 0 or min(ismember(active_cons, active_cons1)) == 1:
        newadd = 1
        success = 1
        if mindis(x, xU)[0] == 0:
            newadd = 0
    else:
        success = 0
        newadd = 0
        xU = np.hstack((xU, x))
    return x, xE, xU, newadd, success


############################################ Plot ############################################
def plot_alpha_dogs(xE, xU, yE, SigmaT, xc_min, yc, yd, funr, num_iter, K, L, Nm):
    '''
    This function is set for plotting Alpha-DOGS algorithm on toy problem, e.g. Schwefel function, containing:
    continuous search function, discrete search function, regression function, truth function and function evaluation.

    :param xE: Evaluated points.
    :param xU: Support points, useful for building continuous function.
    :param yE: Function evaluation of xE.
    :param SigmaT: Uncertainty on xE.
    :param xc_min: Minimum point of continuous search function at this iteration.
    :param yc: Minimum of continuous search function built by piecewise quadratic model.
    :param yd: Minimum of discrete search function.
    :param funr: Truth function.
    :param num_iter: Number of iteration.
    :param K: Tuning parameter of continuous search function.
    :param L: Tuning parameter of discrete search function.
    :param Nm: Mesh size.
    :return: Plot for each iteration of Alpha-DOGS algorithm, save the fig at 'plot' folder.
    '''
    #####   Generates plots of cs, ds and function evaluation   ######
    n = xE.shape[0]
    K0 = np.ptp(yE, axis=0)
    #####  Plot the truth function  #####
    plt.figure()
    axes = plt.gca()
    axes.set_ylim([-4, 4])
    xall = np.arange(-0.05, 1.01, 0.001)
    yall = np.arange(-0.05, 1.01, 0.001)
    for i in range(len(xall)):
        yall[i] = funr(np.array([xall[i]]))
    plt.plot(xall, yall, 'k-', label='Truth function')
    inter_par = Inter_par(method="NPS")
    ##### Plot the discrete search function  #####
    [interpo_par, yp] = regressionparametarization(xE, yE, SigmaT, inter_par)
    sd = np.amin((yp, 2 * yE - yp), 0) - L * SigmaT
    plt.scatter(xE[0], sd, color='r', marker='s', s=15, label='Discrete search function')
    ##### Plot the continuous search function  #####
    xi = np.hstack([xE, xU])
    sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
    tri = np.zeros((xi.shape[1] - 1, 2))
    tri[:, 0] = sx[:xi.shape[1] - 1]
    tri[:, 1] = sx[1:]
    tri = tri.astype(np.int32)
    Sc = np.array([])

    K0 = np.ptp(yE, axis=0)
    for ii in range(len(tri)):
        temp_x = xi[:, tri[ii, :]]
        x = np.arange(temp_x[0, 0], temp_x[0, 1]+0.005, 0.005)
        temp_Sc = np.zeros(len(x))
        R2, xc = circhyp(xi[:, tri[ii, :]], n)
        for jj in range(len(x)):
            temp_Sc[jj] = interpolate_val(x[jj], inter_par) - K * K0 * (R2 - np.linalg.norm(x[jj] - xc) ** 2)
        Sc = np.hstack([Sc, temp_Sc])
    x_sc = np.linspace(0, np.max(xi), len(Sc))
    plt.plot(x_sc, Sc, 'g--', label='Continuous search function')
    #####  Plot the separable point for continuous search function  #####
    sscc = np.zeros(xE.shape[1])
    for j in range(len(sscc)):
        for i in range(len(x_sc)):
            if norm(x_sc[i] - xE[0, j]) < 6*1e-3:
                sscc[j] = Sc[i]
    plt.scatter(xE[0], sscc, color='green', s=10)
    #####  Plot the errorbar  #####
    plt.errorbar(xE[0], yE, yerr=SigmaT, fmt='o', label='Function evaluation')
    ########    plot the regression function   ########
    yrall = np.arange(-0.05, 1.01, 0.001)
    for i in range(len(yrall)):
        yrall[i] = interpolate_val(xall[i], inter_par)
    plt.plot(xall, yrall, 'b--', label='Regression function')

    if mindis(xc_min, xE)[0] < 1e-6:
        plt.annotate('Mesh Refinement Iteration', xy=(0.6, 3.5), fontsize=8)
    else:
        if yc < yd:
            #####  Plot the minimum of continuous search function as star  #####
            plt.scatter(xc_min, np.min(Sc), marker=(5, 2))
            #####  Plot the arrowhead  point at minimum of continuous search funtion  #####
            plt.annotate('Identifying sampling', xy=(xc_min, np.min(Sc)),
                         xytext=(np.abs(xc_min-0.5), np.min(Sc)-1),
                                arrowprops=dict(facecolor='black', shrink=0.05))

        else:
            #####  Plot the arrowhead  point at minimum of discrete search funtion  #####
            xd = xE[:, np.argmin(sd)]
            plt.annotate('Supplemental Sampling', xy=(xd[0], np.min(sd)), xytext=(np.abs(xd[0]-0.5), np.min(sd)-1),
                         arrowprops=dict(facecolor='black', shrink=0.05))
    ##### Plot the information about parameters of each iteration  #####
    plt.annotate('#Iterations = ' + str(int(num_iter)), xy=(0, 3.5), fontsize=8)
    plt.annotate('Mesh size = ' + str(int(Nm)), xy=(0.35, 3.5), fontsize=8, color='b')
    plt.annotate('K = ' + str(int(K)), xy=(0.35, 3), fontsize=8, color='b')
    plt.annotate('Range(Y) = ' + str(float("{0:.4f}".format(K0))), xy=(0.28, 2.5), fontsize=8, color='b')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", prop={'size': 6}, borderaxespad=0.)
    plt.grid()
    ##### Check if the folder 'plot' exists or not  #####
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path + "/plot/cd_movie"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    # Save fig
    plt.savefig('plot/cd_movie/cd_movie' + str(num_iter) +'.png', format='png', dpi=1000)
    return


def plot_delta_dogs(xE, yE, alg, sc, Kini, fun_arg, p_iter, r_ind, num_ini, Nm, Nm_p1, ff):
    '''
    This function is set for plotting Delta-DOGS algorithm or Delta-DOGS with DR on toy problem, e.g. Schwefel function or Schwefel + Quadratic.
    Also this function can plot continuous search function for cosntantK and Adaptive K. Parameters containing:
    continuous search function, discrete search function, regression function, truth function and function evaluation.

    :param xE: The data of original input, evaluated points.
    :param yE: Function evaluation of xE.
    :param alg: The algorithm that we are using: 'DR' represents Dimension Reduction, 'Ddogs' represents regular Delta-DOGS.
    :param sc: Type of continuous search function, can be 'ConstantK' or 'AdaptiveK'.
    :param Kini: The initial tuning parameter for ConstantK search function.
    :param fun_arg: Type of truth function.
    :param p_iter: Set for DimRec to represent which points is for 1D random search.
    :param r_ind: Represents which coordination is reduced.
    :param num_ini: Represents the number of initial evaluated points.
    :param Nm: Current mesh size.
    :param Nm_p1: Represents the number of mesh refinement at 1D reduced model.
    :return: Plot for each iteration of Delta-DOGS algorithm, save the fig at 'plot/DimRec' folder.
    '''
    # Plot the truth function
    n = xE.shape[0]
    p = len(r_ind)
    plt.figure()
    x = y = np.linspace(-0.05, 1.05, 500)
    X, Y = np.meshgrid(x, y)
    if fun_arg == 2: # Schwefel
        Z = (-np.multiply(X, np.sin(np.sqrt(abs(500*X)))) - np.multiply(Y, np.sin(np.sqrt(abs(500*Y)))))/2
        y0 = - 1.6759 * n
        xmin = 0.8419 * np.ones((n))

    elif fun_arg == 5: # Schwefel + Quadratic
        Z = - np.multiply(X/2, np.sin(np.sqrt(abs(500*X)))) + 10 * (Y - 0.92) ** 2
        xmin = np.array([0.89536, 0.94188])

    elif fun_arg == 6: # Griewank
        Z = 1 + 1/4 * ((X-0.67) ** 2 + (Y-0.21) ** 2) - np.cos(X) * np.cos(Y/np.sqrt(2))
        y0 = 0.08026
        xmin = np.array([0.21875, 0.09375])

    elif fun_arg == 7: # Shubert
        s1 = 0
        s2 = 0
        for i in range(5):
            s1 += (i+1) * np.cos((i+2) * (X-0.45) + (i+1))
            s2 += (i+1) * np.cos((i+2) * (Y-0.45) + (i+1))
        Z = s1 * s2
        y0 = -32.7533
        xmin = np.array([0.78125, 0.25])

    plt.contourf(X, Y, Z, cmap='gray')
    plt.colorbar()
    # Plot the initial points.
    plt.scatter(xE[0, :num_ini], xE[1, :num_ini], c='w', label='Initial points', s=10)
    # Plot the rest point.
    plt.scatter(xE[0, num_ini:], xE[1, num_ini:], c='b', label='Other points', s=10)
    # Plot the random search point.
    if alg == "DR":
        if sc == "ConstantK":
            plt.title(r"$\Delta$-DOGS(Z): " + sc + ': K = ' + str(Kini) + ' ' + str(len(p_iter)) + "th Iteration: RD = " + str(p) + ', MS = ' + str(Nm), y=1.05)
        elif sc == "AdaptiveK":
            plt.title(r"$\Delta$-DOGS(Z): " + sc + ' ' + str(len(p_iter)) + "th Iteration: RD = " + str(p) + ', MS = ' + str(Nm), y=1.05)
        # Represents which coordinaion is performing random search
#        if sum(r_ind+1) == 1:
#            plt.plot(np.zeros(len(y)), y, c='r')
#        elif sum(r_ind+1) == 2:
#            plt.plot(x, np.zeros(len(x)), c='r')
        if len(p_iter) != 1:
            if max(p_iter) == 1:
                ind = len(p_iter) - Nm_p1
            else:
                ind = np.argmax(p_iter) - Nm_p1
            plt.scatter(xE[0, num_ini:ind+num_ini], xE[1, num_ini:ind+num_ini], c='g', label='1D-Reduced',s=10)
    else:
        if sc == "ConstantK":
            plt.title(r"$\Delta$-DOGS(Z) " + sc + ': K = ' + str(Kini) + ' ' + str(len(p_iter)) + "th Iteration: " + 'MeshSize = ' + str(Nm), y=1.05)
        elif sc == "AdaptiveK":
            plt.title(r"$\Delta$-DOGS(Z): " + sc + ' ' + str(len(p_iter)) + "th Iteration: " + 'MeshSize = ' + str(Nm), y=1.05)

    # Plot the latest point.
    plt.scatter(xE[0, -1], xE[1, -1], c='r', label='Current Evaluate point', s=10)
    # Plot the reduced regression model
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(xmin[0], xmin[1], marker=(5, 2), label='Global minimum')
    plt.grid()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=5, prop={'size': 6}, borderaxespad=0.)
    ##### Check if the folder 'plot' exists or not  #####
    if len(p_iter) == 1:
        current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        plot_folder = current_path + "/plot/DimRed" + '/' + str(ff)
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
    # Save fig
    plt.savefig('plot/DimRed/' + str(ff) + '/DR' + str(len(p_iter)) +'.png', format='png', dpi=1000)
    return


#########################  Dimension Reduction  ##########################################

# Sensitivity Analysis
def sensitivity_analysis(x, y, p):
    n = x.shape[0]
    # ssf is the sum of squares due to the factor \hat{x_j}
    ssf = np.zeros(n)
    for i in range(n):
        u, indices = np.unique(x[i], return_inverse=True)
        sim_j = np.array([ jj == x[i] for jj in u])
        ssf[i] = np.sum(np.sum(sim_j, axis=1) * (np.array([ np.mean(y[sim_j[i]]) for i in range(sim_j.shape[0]) ]) - np.mean(y))**2)
    ssf /= np.sum((y-np.mean(y))**2)
    index = ssf.argsort()[-p:][::-1]
    index = np.sort(index)
    return index


def SQ_gradient(x):
    x = x.reshape(-1, 1)
    dx = np.zeros((2, 1))
    dx[0][0] = -1 / 2 * np.sin(np.abs(500*x[0][0])) - 250 * x[0][0] * np.cos(np.abs(500*x[0][0]))
    dx[1][0] = 20 * (x[1][0] - 0.92)
    return dx


# Active Subspace
def gradient_analysis_svd(xE, Nm):
    n = xE.shape[0]
    m = xE.shape[1]
    G = np.zeros((n, m))
    for i in range(m):
        G[:, i] = SQ_gradient(xE[:, i]).T[0]
    G = np.dot(G, G.T)
    s, W = np.linalg.eig(G)
    xas = np.dot(W[:, 0], xE).reshape(-1, 1).T
    W1 = W[:, 0].reshape(-1, 1)
    xas = np.round(xas * Nm) / Nm
    xas_unique = xas[:, 0].reshape(-1, 1)
    dup_ind = np.array([])
    for i in range(1, m):
        for j in range(xas_unique.shape[1]):
            if (xas[:, i] == xas_unique[:, j]).all():
                dup_ind = np.hstack((dup_ind, i))
                break
            if j == xas_unique.shape[1] - 1:
                xas_unique = np.hstack((xas_unique, xas[:, i].reshape(-1, 1)))
    return xas_unique, W1, dup_ind


def svd_inv(xc, W1):
    x0 = np.ones((2, 1))
    n = 2
    costfun = lambda x: svd_inv_cost(x, xc, W1)[0]
    costjac = lambda x: svd_inv_cost(x, xc, W1)[1]
    opt = {'disp': False}
    bnds = tuple([(0, 1) for i in range(int(n))])
    res = optimize.minimize(costfun, x0, jac=costjac, method='TNC', bounds=bnds, options=opt)
    x = res.x
    y = res.fun
    return x, y


def svd_inv_cost(x, xc, W1):
    x = x.reshape(-1, 1)
    M = (np.dot(W1.T[0], x) - xc) ** 2
    DM = 2 * (np.dot(W1.T[0], x) - xc) * W1
    return M, DM.T


def reduced_polyharm_parametrization(xi, yi, y0):
    errr = 1000
    dimr = 1
    err = np.zeros(xi.shape[0])
    for dim in range(xi.shape[0]):
        xr = np.delete(xi, dim, 0)
        regr_par, err[dim], regret, indr = reduced_parameterization(xr, yi)
        if err[dim] < errr:
            errr = err[dim]
            dimr = dim
            regr_par_r = regr_par
    return regr_par_r, err, regret, dimr


# New method of data determination for DR
def reduced_parameterization(xr, yi, y0):
    xr_uni, yi_uni, index = data_merge(xr, yi)
    inter_par_r = Inter_par(method='NPS')
    inter_par_r = interpolateparameterization(xr_uni, yi_uni, inter_par_r)
    e0 = 100
    yr = np.zeros(xr_uni.shape[1])
    error = np.zeros(xr_uni.shape[1])
    regret = np.zeros(xr_uni.shape[1])
    for j in range(xr_uni.shape[1]):
        yr[j] = interpolate_val(xr_uni[:, j], inter_par_r)
        error[j] = yr[j] - yi_uni[j]
        regret[j] = yr[j] - yr[j]
        if error[j] < e0:
            regr_par = inter_par_r
            err = error[j]
            indr = j
    regret = min(yr) - y0
    return regr_par, err, regret, indr


def active_subspace_direction(xE, inter_par, Nm, k):
    '''
    :param xE: The data of original input, evaluated points.
    :param inter_par: Parameter of interpolation.
    :param Nm: Mesh Size.
    :param alpha: The oversampling factor, between 2 and 10.
    :param k: The dimension of reduced model, determined by previous iteration, represented by p
              in the previous code.
    '''
    n = xE.shape[0]
    C = np.zeros((n, n))
    # Perform uniform sampling
    # M is the number of gradient sample of interpolation
    alpha = 1
    # Indicator represents that the eigenvalues of reduced space satisfy the threshold on the magnitude
    # of the eigenvalues.
    indicator = 0
    while 1:
        alpha += 1
        M = np.int(alpha * k * np.log(n))
        for i in range(M):
            # rp represents random point
            rp = np.random.rand(n, 1)
            rp = np.round(rp * Nm) / Nm
            rp_g = interpolate_grad(rp, inter_par)
            C += np.dot(rp_g, rp_g.T)
        C /= M
        # Identify the eigenvector - active dimension
        U, s, V = np.linalg.svd(C, full_matrices=True)
        expper = sum(s[:k])/sum(s)
        if expper >= k / n:
            # The eigenvectors returned by U have already been normalized.
            # reddir is the first k eigenvectors of C, the rotation matrix for reduced space.
            reddir = U[:, :k]
            indicator = 1
            break
        else:
            continue
        if alpha == 10:
            break
    return reddir, expper, indicator


def tringulation_search_bound_reduced_dim(reddir, inter_par, xi, y0, K0, ind_min):
    inf = 1e+20
    n = xi.shape[0]
#    xm, ym = inter_min(xi[:, ind_min], inter_par)
#    ym = ym[0, 0]  # If using scipy package, ym would first be a two dimensions array.
#    if ym > y0:
#        ym = inf
    # construct Deluanay tringulation
    if n == 1:
        sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
        tri = np.zeros((xi.shape[1] - 1, 2))
        tri[:, 0] = sx[:xi.shape[1] - 1]
        tri[:, 1] = sx[1:]
        tri = tri.astype(np.int32)
    else:
        options = 'Qt Qbb Qc' if n <= 3 else 'Qt Qbb Qc Qx'
        tri = scipy.spatial.Delaunay(xi.T, qhull_options=options).simplices
        keep = np.ones(len(tri), dtype=bool)
        for i, t in enumerate(tri):
            if abs(np.linalg.det(np.hstack((xi.T[t], np.ones([1, n + 1]).T)))) < 1E-15:
                keep[i] = False  # Point is coplanar, we don't want to keep it
        tri = tri[keep]
    # Sc contains the continuous search function value of the center of each Delaunay simplex
    Sc = np.zeros([np.shape(tri)[0]])
    Scl = np.zeros([np.shape(tri)[0]])
    for ii in range(np.shape(tri)[0]):
        R2, xc = circhyp(xi[:, tri[ii, :]], n)
        # if R2 != np.inf:
        if R2 < inf:
            # initialze with body center of each simplex
            x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
            Sc[ii] = (interpolate_val(x, inter_par) - y0) / (R2 - np.linalg.norm(x - xc) ** 2)
            if np.sum(ind_min == tri[ii, :]):
                Scl[ii] = np.copy(Sc[ii])
            else:
                Scl[ii] = inf
        else:
            Scl[ii] = inf
            Sc[ii] = inf

    # Global one, the minimum of Sc has the minimum value of all circumcenters.
    ind = np.argmin(Sc)
    R2, xc = circhyp(xi[:, tri[ind, :]], n)
    # x is the center of this
    x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))
    xm, ym = Adoptive_K_Search_reduced_dim(x, reddir, inter_par, xc, R2, y0, K0)
    # Local one
    ind = np.argmin(Scl)
    R2, xc = circhyp(xi[:, tri[ind, :]], n)
    # Notice!! ind_min may have a problen as an index
    x = np.copy(xi[:, ind_min].reshape(-1, 1))
    xml, yml = Adoptive_K_Search_reduced_dim(x, reddir, inter_par, xc, R2, y0, K0)
    if yml < 2 * ym:
        xm = np.copy(xml)
        ym = np.copy(yml)
    xm = xm.reshape(-1, 1)
    ym = - 1 / ym
    return xm, ym


def Adoptive_K_Search_reduced_dim(x0, reddir, inter_par, xc, R2, y0, K0, lb=[], ub=[]):
    # Find the minimizer of the search fucntion in a simplex
    n = x0.shape[0]
    costfun = lambda x: AdaptiveK_search_cost_reduced_dim(x, reddir, inter_par, xc, R2, y0, K0)[0]
    costjac = lambda x: AdaptiveK_search_cost_reduced_dim(x, reddir, inter_par, xc, R2, y0, K0)[1]
    opt = {'disp': False}
    bnds = tuple([(0, 1) for i in range(int(n))])
    # res = optimize.minimize(costfun, x0, jac=costjac, method='TNC', bounds=bnds, options=opt)
    res = optimize.minimize(costfun, x0, jac=costjac, method='TNC', bounds=bnds, options=opt)
    x = res.x
    y = res.fun
    return x, y


def AdaptiveK_search_cost_reduced_dim(x, reddir, inter_par, xc, R2, y0, K0):
    x = x.reshape(-1, 1)
    # TODO: Solve interpolate_value!!!!
    p = interpolate_val(x, inter_par)
    NR2 = find_R2_linear_transform_Delaunay(reddir, xc, R2)
    e = NR2 - np.linalg.norm(np.dot(reddir.T, x - xc)) ** 2
    # Search function value
    M = - e * K0 / (p - y0)
    # M = (p - y0) / e
    gp = interpolate_grad(x, inter_par)
    ge = - 2 * np.dot((x - xc).T, np.dot(reddir, reddir.T)).T
    DM = - ge * K0 / (p - y0) + K0 * e * gp / (p - y0) ** 2
    return M, DM.T


def find_R2_linear_transform_Delaunay(reddir, xc, R2):
    xc = xc.reshape(-1, 1)
    # Line equation:
    #    x_i = c_i + t * w_i, i from 1 to n.
    #    x_i: each coordinate
    #    c_i: circumcenter
    #    w_i: eigenvector direction
    # Sphere equation: sum[(x_i - c_i)**2] = R**2
    # Substitue line equation into sphere, solve for t
    t = np.sqrt(R2 / sum(reddir**2))
    # Find the intersection on original circumcenter, I1 and I2
    I1 = xc + t * reddir
    I2 = xc - t * reddir
    # RI1 and RI2 represents the intersection in reduced model
    RI1 = np.dot(reddir.T, I1)
    RI2 = np.dot(reddir.T, I2)
    NR2 = np.linalg.norm(RI2 - RI1) ** 2
    return NR2


def find_boundary_reduced_model(reddir):
    n = reddir.shape[0]
    m = reddir.shape[1]
    # The first row of reduced_bound is the minimum, second row - maximum.
    # The first column of reduced_bound is from the first reduced dimension, etc.
    reduced_bound = np.zeros((2, m))
    minimizer = np.zeros((n, m))
    maximizer = np.zeros((n, m))
    bnds = tuple([(0, 1) for i in range(int(n))])
    opt = {'disp':False}
    for i in range(m):
        W = reddir[:, i].reshape(-1, 1)
        min_res = optimize.linprog(W.T[0], bounds=bnds, options=opt)
        max_res = optimize.linprog(-W.T[0], bounds=bnds, options=opt)
        reduced_bound[0][i] = min_res.fun
        reduced_bound[1][i] = -max_res.fun
        minimizer[:, i] = min_res.x.reshape(-1, 1)[:, 0]
        maximizer[:, i] = max_res.x.reshape(-1, 1)[:, 0]
    return reduced_bound, minimizer, maximizer


############################################ Test Examples ############################################
def read_str(S, judge):
    '''
    This function is designed for reading input from 'dat' file.
    :param S: Input string has the form: string=numbers.
    :param judge: judge == 'i', means int, judge == 'f' means float.
    :return: Input numbers .
    '''
    s = S[0]
    for i in range(len(s)):
        if s[i] != '=':
            continue
        elif s[i] == '=':
            i += 1
            break
    r = s[i:]
    if judge == 'i':
        r = int(r)
    else:
        r = float(r)
    return r


def solver_lorenz_alpha_DOGS_and_X(x, t, h, alg):
    '''
    This Solver is set for implementing Alpha-DOGSX algorithmon Lorenz system.
    :param x: Point of interest. This point is in physical bound.
    :param T: Total attractor time at current point.
    :param h: Step size
    :param alg: The optimization method, should be "alpha_dogs" or "alpha_dogsx".
    :return: Function evaluation at x, uncertainty sig at x.
    '''
    x1 = x.reshape(-1, 1)
    n = len(x1)                                     # n represents the dimension of data

    var_opt = io.loadmat("allpoints/pre_opt")
    idx = var_opt['num_point'][0, 0]
    flag = var_opt['flag'][0, 0]
    DT = 10                                        # Time length increment for alpha_dogs
    time_method = 1                                # RK4
    sigma0 = 3

    # sigma0 = 3
    # sig = sigma0 / np.sqrt(t) + 0.8 * h ** 3

    if n == 1:
        y0 = np.array([23.5712])
    elif n == 2:
        y0 = np.array([23.5712, 8.6107])

    if flag != 2:
        if flag == 1:   # flag 1 : new point, alpha_dogs is the same with alpha_dogsx
            J, zs, ys, xs = lorenz.lorenz_lost2(x, t, h, y0, time_method)

        elif flag == 0:  # flag = 0: existing point
            data = io.loadmat("allpoints/pt_to_eval" + str(idx) + ".mat")
            T_zs_lorenz = data['T'][0, 0]
            h_pre = data['h'][0, 0]
            if np.abs(h - h_pre) < 1e-5:
                if alg == "alpha_dogs":
                    t = T_zs_lorenz + DT
#                elif alg == "alpha_dogsx":
#                    t += T_zs_lorenz
                J, zs, ys, xs = lorenz.lorenz_lost2(x, t, h, y0, time_method, idx)
            elif h != h_pre:
                J, zs, ys, xs = lorenz.lorenz_lost2(x, t, h, y0, time_method)

        if n == 1:
            # First way to calculate sigma, using UQ from Shahrouz
            length = int(min((t/h)/5, 40))
            xx = uq.data_moving_average(zs, length).values
            sig = np.sqrt(uq.stationary_statistical_learning_reduced(xx, 18)[0])  # Work for 1D Lorenz.

            # Second way to calculate sigma
#            sig = sigma0 / np.sqrt(t) + 0.8 * h ** 3
        elif n == 2:
            # sig = sigma0 / np.sqrt(t/h) + 0.8 * h ** 3
            sig = sigma0 / np.sqrt(t) + 0.8 * h ** 3  # Temporary working for 2D Lorenz, without theory.
            # sig = uq.statistical_std(zs)  # Not working.

        fout = {'zs': zs, 'ys': ys, 'xs': xs, 'h': h, 'T': t, 'J': J}
        io.savemat("allpoints/pt_to_eval" + str(idx) + ".mat", fout)

        return J, sig, t, h
    else:  # flag == 2, no function evaluation.
        return


def fit_sigma0():
    data = io.loadmat("STD_VS_T.mat")
    T = data["T"]
    std = data["STD"]
    return np.exp(np.mean(np.log(std) + 1 / 2 * np.log(T)))


def alpha_X_mesh_sampling(d_sigma, sigma, T0, h0, sigma0):
    DT = 10
    C0 = 0.8
    cost_min = 1e+10
    h1 = h0 / 2
    T1 = np.ceil( ( sigma0 / (sigma-d_sigma-C0*h1**3) )**2 )
    cost_l = T1 / h1
    if cost_l < cost_min:
        cost_min = cost_l
        hnew = h1
        Tnew = T1
    else:
        d_sigma /= 2
    for l in range(10):
        if cost_min > 1e+10:   # Computation cost too large, divide the required delta sigma by 2.
            d_sigma /= 2
        else:
            h1 /= 2            # Computation cost is moderate, try another time step length h.
        T1 = np.ceil( ( sigma0 / (sigma-d_sigma-C0*h1**3) )**2 )
        cost_l = T1 / h1
#            else:
#                cost_l = (T1 - T0) / h1
        if cost_l < cost_min and h1 > 5e-3 - 1e-6:
            cost_min = cost_l
            hnew = h1
            Tnew = T1
        if cost_min > 1e+5:
            d_sigma /= 2
        print('=============================')
        print('d_sigma = ', d_sigma)
        print('l = ', l, 'h = ', h1, 'T1 = ', T1)
        print('Tnew = ', Tnew, 'hnew = ', hnew)
        print('cost_l = ', cost_l, 'cost_min = ', cost_min)
    if h0 < 5e-4:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('h0 < 5e-4')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        hnew = 0.005
    if cost_min > 1e+10:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('cost_min > 1e+10')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        Tnew =T0 + 10 * DT
    if h0 < 5e-4:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('h0 < 5e-4')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        hnew = 0.005
    cost_min = Tnew/hnew
    print('============  Summary  =================')
    print('cost_min = ', cost_min)
    print('cost_l = ', cost_l)
    print('Tnew = ', Tnew)
    print('hnew = ', hnew)
    return Tnew, hnew


def fe_schwefel(x, h, T, sigma0, C0, p, yp=0, Tp=0):
    '''
    Define the function evaluation of schwefel toy problem based on alpha-DOGSX algorithm.
    :param x: Arguments of function evaluation.
    :param h: The time step h used for generating synthetic discretization error.
    return : Function evaluation containing sample error and discretization error, and 2 kinds of errors.
    '''
    sum_error = 0
    for i in range(T):
        sum_error += sigma0 * np.random.randn()
    y = -sum(np.multiply(500 * x, np.sin(np.sqrt(abs(500 * x))))) / 250 + (yp * Tp + sum_error) / (T + Tp)
    # sr: sampling error
    sr = sigma0 / np.sqrt(T + Tp)
    # dr: discretization error
    dr = C0 * h**p
    sig = sr + dr
    return y, sig

#############################           LORENZ             ##################################################
def normalize_bounds(x0, lb, ub):
    n = len(lb)  # n represents dimensions
    m = x0.shape[1]  # m represents the number of sample data
    x = np.copy(x0)
    for i in range(n):
        for j in range(m):
            x[i][j] = (x[i][j] - lb[i]) / (ub[i] - lb[i])
    return x


def physical_bounds(x0, lb, ub):
    '''
    :param x0: normalized point
    :param lb: real lower bound
    :param ub: real upper bound
    :return: physical scale of the point
    '''
    n = len(lb)  # n represents dimensions
    try:
        m = x0.shape[1]  # m represents the number of sample data
    except:
        m = x0.shape[0]
    x = np.copy(x0)
    for i in range(n):
        for j in range(m):
            x[i][j] = (x[i][j])*(ub[i] - lb[i]) + lb[i]

    return x


#########################  Check for the time of generating files ##########################################
def creation_date(path_to_file):    # path_to_file is the name without Path().
    if platform.system() == 'Windows':  # Check the system, if windows use this line.
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:  # This is for linux system.
            return stat.st_mtime


def function_evaluation_sign():
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    pts_path = current_path + "/allpoints/pts_to_eval.dat"
    surr_path = current_path + "/allpoints/surr_J_new.dat"
    pts_time = creation_date(pts_path)
    surr_time = creation_date(surr_path)
    if pts_time < surr_time:  # This means that function evaluation is successful.
        sign = 1
    elif pts_time >= surr_time:
        sign = 0
    return sign

#################  The solver function designed for lorenze system  ##################


def solver_lorenz():  # flag = 1 : new point
    var_opt = io.loadmat("allpoints/pre_opt_IC")
    bnd1 = var_opt['lb'][0]
    bnd2 = var_opt['ub'][0]
    n = var_opt['n'][0, 0]
    T_lorenz = 5
    h = 0.005
    if n == 1:
        y0 = np.array([23.5712])
    elif n == 2:
        y0 = np.array([23.5712, 23.5712])
    elif n == 3:
        y0 = np.array([23.5712, 23.5712, 23.5712])
    time_method = 1
    DT = 10

    fin = open("allpoints/pts_to_eval.dat", "r")
    flag = read_str(fin.readline().split(), 'i')
    idx = read_str(fin.readline().split(), 'i')
    xm = np.zeros(n)
    for i in range(n):
        xm[i] = read_str(fin.readline().split(), 'f')
    fin.close()

    if flag != 2:
        if flag == 1:  # flag = 1: new point
            T = T_lorenz
            J, zs, ys, xs = lorenz.lorenz_lost2(xm, T, h, y0, time_method)
        elif flag == 0:  # flag = 0: existing point
            data = io.loadmat("allpoints/pt_to_eval" + str(idx) + ".mat")
            T_zs_lorenz = data['T'][0, 0]
            T = T_zs_lorenz + DT
            J, zs, ys, xs = lorenz.lorenz_lost2(xm, T, h, y0, time_method, idx)

        fout_surr = open("allpoints/surr_J_new.dat", "w")
        for i in range(zs.shape[0]):
            fout_surr.write(str((zs[i] - y0)[0]) + "\n")
        fout_surr.close()

        fout = {'zs': zs, 'ys': ys, 'xs': xs, 'h': h, 'T': T, 'J': J}
        io.savemat("allpoints/pt_to_eval" + str(idx) + ".mat", fout)

        return

    else:  # flag = 2, this is mesh refinement iteration, no function evaluation is needed.

        return
#################  The alpha-DOGS algprithm for lorenz system  ##################


def DOGS_standalone_IC():
    '''
    This function reads the set of evaluated points and writes them into the desired file to perform function evaluations
    Note: DOGS_standalone() only exists at the inactivated iterations.
    :return: points that needs to be evaluated
    '''
    # For future debugging, remind that xc and xd generate by DOGS_standalone() is set to be a one dimension row vector.
    # While lb and ub should be a two dimension matrix, i.e. a column vector.
    # The following lines will read input from 'pre_opt_IC' file:
    var_opt = io.loadmat("allpoints/pre_opt_IC")
    n = var_opt['n'][0, 0]
    K = var_opt['K'][0, 0]
    L = var_opt['L'][0, 0]
    Nm = var_opt['Nm'][0, 0]
    bnd2 = var_opt['ub'][0]
    bnd1 = var_opt['lb'][0]
    lb = np.zeros(n)
    ub = np.ones(n)
    user = var_opt['user'][0]
    idx = var_opt['num_point'][0, 0]
    flag = var_opt['flag'][0, 0]
    method = var_opt['inter_par_method']
    xE = var_opt['xE']
    xU = var_opt['xU']
    k = var_opt['iter'][0, 0]
    iter_max = var_opt['iter_max'][0, 0]
    # fe_times = var_opt['fe_times']


    if xU.shape[1] == 0:
        xU = xU.reshape(n, 0)

    Data = io.loadmat("allpoints/Yall")
    yE = Data['yE'][0]
    SigmaT = Data['SigmaT'][0]
    T = Data['T'][0]

    # identify whether or not the function evaluation is successful:
    sign = function_evaluation_sign()
    # Read the result from 'surr_J_new.dat' file that generated by solver_lorenz:
    if k != 1:
        # if sign == 1:
        zs = np.loadtxt("allpoints/surr_J_new.dat")
        xx = uq.data_moving_average(zs, 40).values
        ind = tr.transient_removal(xx)
        sig = np.sqrt(uq.stationary_statistical_learning_reduced(xx[ind:], 18)[0])
        J = np.abs(np.mean(xx[ind:]))

        if flag == 1:  # New point

            yE = np.hstack([yE, J])
            SigmaT = np.hstack([SigmaT, sig])
            T = np.hstack([T, len(zs)])

        elif flag == 0:  # existing point

            yE[idx] = J
            SigmaT[idx] = sig
            T[idx] = len(zs)

    #############################################################################
    # The following only for displaying information.
    # NOTICE : Deleting following lines won't cause any affect.
    print('========================  Iteration = ', k, '=======================================')
    print('point to evaluate at this iteration, x = ', xE[:, idx], "flag = ", flag)
    print('==== flag 1 represents new point, 0 represents existed point  =====')
    print('Function Evaluation at this iter: y = ', yE[idx] + SigmaT[idx])
    print('Minimum of all data points(yE + SigmaT): min = ', np.min(yE + SigmaT))
    print('argmin: x_min = ', xE[:, np.argmin(yE + SigmaT)])
    print('Mesh size = ', Nm)
    #############################################################################
    # Normalize the bounds of xE and xU
    xE = normalize_bounds(xE, bnd1, bnd2)
    xU = normalize_bounds(xU, bnd1, bnd2)

    Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
    Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)
    # Calculate the Regression Function
    inter_par = Inter_par(method=method)
    [inter_par, yp] = regressionparametarization(xE, yE, SigmaT, inter_par)
    K0 = 20  # K0 = np.ptp(yE, axis=0)

    # Calculate the discrete function.
    ind_out = np.argmin(yp + SigmaT)
    sd = np.amin((yp, 2 * yE - yp), 0) - L * SigmaT

    ind_min = np.argmin(yp + SigmaT)

    yd = np.amin(sd)
    ind_exist = np.argmin(sd)

    xd = xE[:, ind_exist]

    if ind_min != ind_min:
        # yE[ind_exist] = ((fun(xd)) + yE[ind_exist] * T[ind_exist]) / (T[ind_exist] + 1)
        # T[ind_exist] = T[ind_exist] + 1

        return
    else:

        if SigmaT[ind_exist] < 0.01 * np.ptp(yE, axis=0) * (np.max(ub - lb)) / Nm:
            yd = np.inf

        # Calcuate the unevaluated support points:
        yu = np.zeros([1, xU.shape[1]])
        if xU.shape[1] != 0:
            for ii in range(xU.shape[1]):
                tmp = interpolate_val(xU[:, ii], inter_par) - np.amin(yp)
                yu[0, ii] = tmp / mindis(xU[:, ii], xE)[0]

        if xU.shape[1] != 0 and np.amin(yu) < 0:
            ind = np.argmin(yu)
            xc = np.copy(xU[:, ind])
            yc = -np.inf
            xU = scipy.delete(xU, ind, 1)  # delete the minimum element in xU, which is going to be incorporated in xE
        else:
            while 1:
                xs, ind_min = add_sup(xE, xU, ind_min)
                xc, yc = tringulation_search_bound_constantK(inter_par, xs, K * K0, ind_min)
                yc = yc[0, 0]
                if interpolate_val(xc, inter_par) < min(yp):
                    xc = np.round(xc * Nm) / Nm
                    break

                else:
                    xc = np.round(xc * Nm) / Nm
                    if mindis(xc, xE)[0] < 1e-6:
                        break
                    xc, xE, xU, success, _ = points_neighbers_find(xc, xE, xU, Bin, Ain)
                    xc = xc.T[0]
                    if success == 1:
                        break
                    else:
                        yu = np.hstack([yu, (interpolate_val(xc, inter_par) - min(yp)) / mindis(xc, xE)[0]])

            if xU.shape[1] != 0 and mindis(xc, xE)[0] > 1e-10:
                tmp = (interpolate_val(xc, inter_par) - min(yp)) / mindis(xc, xE)[0]
                if 3*np.amin(yu) < tmp:
                    ind = np.argmin(yu)
                    xc = np.copy(xU[:, ind])
                    yc = -np.inf
                    xU = scipy.delete(xU, ind, 1)  # delete the minimum element in xU, which is incorporated in xE
        # Generate the stop file at this iteration:
        if k + 1 <= iter_max:
            stop = 0
        elif k + 1 > iter_max:
            stop = 1

        fout = open("allpoints/stop.dat", 'w')
        fout.write(str(stop) + "\n")
        fout.close()

        # MESH REFINEMENT ITERATION:
        if mindis(xc, xE)[0] < 1e-6:
            K = 2 * K
            Nm = 2 * Nm
            L += 1
            flag = 2  # flag = 2 represents mesh refinement, in this step we don't have function evaluation.

            # Reconstruct the physical bound of xE and xU
            xE = physical_bounds(xE, bnd1, bnd2)
            xU = physical_bounds(xU, bnd1, bnd2)

            # Store the updated information about iteration to the file 'pre_opt_IC.dat'
            var_opt['K'] = K
            var_opt['Nm'] = Nm
            var_opt['L'] = L
            var_opt['xE'] = xE
            var_opt['xU'] = xU
            var_opt['num_point'] = xE.shape[1] - 1  # Doesn't matter, flag = 2, no function evaluation.
            var_opt['flag'] = flag
            var_opt['iter'] = k + 1
            io.savemat("allpoints/pre_opt_IC", var_opt)

            # Store the function evaluations yE, sigma and time length T:
            data = {'yE': yE, 'SigmaT': SigmaT, 'T': T}
            io.savemat("allpoints/Yall", data)

            # Generate the pts_to_eval file for solver_lorenz
            fout = open("allpoints/pts_to_eval.dat", 'w')
            if user == 'Imperial College':
                keywords = ['Awin', 'lambdain', 'fanglein']
                fout.write(str('flagin') + '=' + str(int(flag)) + "\n")
                fout.write(str('IDin') + '=' + str(int(idx)) + "\n")
                for i in range(n):
                    fout.write(str(keywords[i]) + '=' + str(xc[i]) + "\n")
            fout.close()

            return

        if yc < yd:
            if mindis(xc, xE)[0] > 1e-6:

                xE = np.concatenate([xE, xc.reshape(-1, 1)], axis=1)
                flag = 1  # new point
                idx = xE.shape[1] - 1

                # Reconstruct the physical bound of xE and xU
                xE = physical_bounds(xE, bnd1, bnd2)
                xU = physical_bounds(xU, bnd1, bnd2)
                xc = physical_bounds(xc.reshape(-1, 1), bnd1, bnd2)
                xc = xc.T[0]

                # Store the updated information about iteration to the file 'pre_opt_IC.dat'
                # fe_times = np.hstack((fe_times, 1))
                var_opt['K'] = K
                var_opt['Nm'] = Nm
                var_opt['L'] = L
                var_opt['xE'] = xE
                var_opt['xU'] = xU
                var_opt['num_point'] = idx
                var_opt['flag'] = flag
                var_opt['iter'] = k + 1
                # var_opt['fe_times'] = fe_times
                io.savemat("allpoints/pre_opt_IC", var_opt)

                # Store the function evaluations yE, sigma and time length T:
                data = {'yE': yE, 'SigmaT': SigmaT, 'T': T}
                io.savemat("allpoints/Yall", data)

                # Generate the pts_to_eval file for solver_lorenz
                fout = open("allpoints/pts_to_eval.dat", 'w')
                if user == 'Imperial College':
                    keywords = ['Awin', 'lambdain', 'fanglein']
                    fout.write(str('flagin') + '=' + str(int(flag)) + "\n")
                    fout.write(str('IDin') + '=' + str(int(idx)) + "\n")
                    for i in range(n):
                        fout.write(str(keywords[i]) + '=' + str(xc[i]) + "\n")
                fout.close()

                return
        else:
            if mindis(xd, xE)[0] < 1e-10:

                flag = 0  # existing point

                # Reconstruct the physical bound of xE and xU
                xE = physical_bounds(xE, bnd1, bnd2)
                xU = physical_bounds(xU, bnd1, bnd2)
                xd = physical_bounds(xd.reshape(-1, 1), bnd1, bnd2)
                xd = xd.T[0]

                # Store the updated information about iteration to the file 'pre_opt_IC.dat'
                # fe_times[ind_exist] = 1
                var_opt['K'] = K
                var_opt['Nm'] = Nm
                var_opt['L'] = L
                var_opt['xE'] = xE
                var_opt['xU'] = xU
                var_opt['num_point'] = ind_exist
                var_opt['flag'] = flag
                var_opt['iter'] = k + 1
                # var_opt['fe_times'] = fe_times
                io.savemat("allpoints/pre_opt_IC", var_opt)

                # Store the function evaluations yE, sigma and time length T:
                data = {'yE': yE, 'SigmaT': SigmaT, 'T': T}
                io.savemat("allpoints/Yall", data)

                # Generate the pts_to_eval file for solver_lorenz
                fout = open("allpoints/pts_to_eval.dat", 'w')
                if user == 'Imperial College':
                    keywords = ['Awin', 'lambdain', 'fanglein']
                    fout.write(str('flagin') + '=' + str(int(flag)) + "\n")
                    fout.write(str('IDin') + '=' + str(int(ind_exist)) + "\n")
                    for i in range(n):
                        fout.write(str(keywords[i]) + '=' + str(xd[i]) + "\n")
                fout.close()

                return
###################################  Delta-DOGS ##############################


def Delta_DOGS_standalone():
    '''
    This function reads the set of evaluated points and writes them into the desired file to perform function evaluations
    Note: DOGS_standalone() only exists at the inactivated iterations.
    :return: points that needs to be evaluated
    '''
    # For future debugging, remind that xc and xd generate by DOGS_standalone() is set to be a one dimension row vector.
    # While lb and ub should be a two dimension matrix, i.e. a column vector.
    var_opt = io.loadmat("allpoints/pre_opt")
    n = var_opt['n'][0, 0]
    K = var_opt['K'][0, 0]
    L = var_opt['L'][0, 0]
    Nm = var_opt['Nm'][0, 0]
    bnd2 = var_opt['ub'][0]
    bnd1 = var_opt['lb'][0]
    lb = np.zeros(n)
    ub = np.ones(n)
    user = var_opt['user'][0]
    idx = var_opt['num_point'][0, 0]
    flag = var_opt['flag'][0, 0]
    T_lorenz = var_opt['T_lorenz']
    h = var_opt['h_lorenz']
    method = var_opt['inter_par_method']
    xE = var_opt['xE']
    xU = var_opt['xU']
    if xU.shape[1] == 0:
        xU = xU.reshape(n, 0)

    Data = io.loadmat("allpoints/Yall")
    yE = Data['yE'][0]
    SigmaT = Data['SigmaT'][0]

    Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
    Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)

    # TODO FIXME: nff is deleted
    # regret = np.zeros((nff, iter_max))
    # estimate = np.zeros((nff, iter_max))
    # datalength = np.zeros((nff, iter_max))
    # mesh = np.zeros((nff, iter_max))

    inter_par = Inter_par(method=method)
    [inter_par, yp] = regressionparametarization(xE, yE, SigmaT, inter_par)
    K0 = 20# K0 = np.ptp(yE, axis=0)
    # Calculate the discrete function.
    ind_out = np.argmin(yp + SigmaT)
    sd = np.amin((yp, 2 * yE - yp), 0) - L * SigmaT

    ind_min = np.argmin(yp + SigmaT)

    yd = np.amin(sd)
    ind_exist = np.argmin(sd)

    xd = xE[:, ind_exist]

    if ind_min != ind_min:
        # yE[ind_exist] = ((fun(xd)) + yE[ind_exist] * T[ind_exist]) / (T[ind_exist] + 1)
        # T[ind_exist] = T[ind_exist] + 1

        return
    else:

        # if SigmaT[ind_exist] < 0.01 * np.ptp(yE, axis=0) * (np.max(ub - lb)) / Nm:
        #     yd = np.inf

        # Calcuate the unevaluated function:
        yu = np.zeros([1, xU.shape[1]])
        if xU.shape[1] != 0:
            for ii in range(xU.shape[1]):
                tmp = interpolate_val(xU[:, ii], inter_par) - np.amin(yp)
                yu[0, ii] = tmp / mindis(xU[:, ii], xE)[0]

        if xU.shape[1] != 0 and np.amin(yu) < 0:
            t = np.amin(yu)
            ind = np.argmin(yu)
            xc = np.copy(xU[:, ind])
            yc = -np.inf
            xU = scipy.delete(xU, ind, 1)  # create empty array
        else:
            while 1:
                xc, yc = tringulation_search_bound_constantK(inter_par, np.hstack([xE, xU]), K * K0, ind_min)
                yc = yc[0, 0]
                if interpolate_val(xc, inter_par) < min(yp):
                    xc = np.round(xc * Nm) / Nm
                    break

                else:
                    xc = np.round(xc * Nm) / Nm
                    if mindis(xc, xE)[0] < 1e-6:
                        break
                    xc, xE, xU, success, _ = points_neighbers_find(xc, xE, xU, Bin, Ain)
                    xc = xc.T[0]
                    if success == 1:
                        break
                    else:
                        yu = np.hstack([yu, (interpolate_val(xc, inter_par) - min(yp)) / mindis(xc, xE)[0]])

            if xU.shape[1] != 0:
                tmp = (interpolate_val(xc, inter_par) - min(yp)) / mindis(xc, xE)[0]
                if np.amin(yu) < tmp:
                    ind = np.argmin(yu)
                    xc = np.copy(xU[:, ind])
                    yc = -np.inf
                    xU = scipy.delete(xU, ind, 1)  # create empty array

        if mindis(xc, xE)[0] < 1e-6:
            K = 2 * K
            Nm = 2 * Nm
            L += 1
            flag = 2  # flag = 2 represents mesh refinement, in this step we don't have function evaluation.

            var_opt = {}
            var_opt['n'] = n
            var_opt['K'] = K
            var_opt['Nm'] = Nm
            var_opt['L'] = L
            var_opt['lb'] = bnd1
            var_opt['ub'] = bnd2
            var_opt['user'] = user
            var_opt['inter_par_method'] = method
            var_opt['xE'] = xE
            var_opt['xU'] = xU
            var_opt['num_point'] = xE.shape[1] - 1  # Doesn't matter, flag = 2, no function evaluation.
            var_opt['flag'] = flag
            var_opt['T_lorenz'] = T_lorenz
            var_opt['h_lorenz'] = h
            io.savemat("allpoints/pre_opt", var_opt)

            return

        if yc == yc:
            if mindis(xc, xE)[0] > 1e-6:

                xE = np.concatenate([xE, xc.reshape(-1, 1)], axis=1)
                # xm = lb + (ub - lb) * xc
                flag = 1  # new point

                var_opt = {}
                var_opt['n'] = n
                var_opt['K'] = K
                var_opt['Nm'] = Nm
                var_opt['L'] = L
                var_opt['lb'] = bnd1
                var_opt['ub'] = bnd2
                var_opt['user'] = user
                var_opt['inter_par_method'] = method
                var_opt['xE'] = xE
                var_opt['xU'] = xU
                var_opt['num_point'] = xE.shape[1] - 1
                var_opt['flag'] = flag
                var_opt['T_lorenz'] = T_lorenz
                var_opt['h_lorenz'] = h
                io.savemat("allpoints/pre_opt", var_opt)

                return
        else:
            # xm = lb + (ub - lb) * xd
            if mindis(xd, xE)[0] < 1e-10:

                flag = 0  # existing point

                var_opt = {}
                var_opt['n'] = n
                var_opt['K'] = K
                var_opt['Nm'] = Nm
                var_opt['L'] = L
                var_opt['lb'] = bnd1
                var_opt['ub'] = bnd2
                var_opt['user'] = user
                var_opt['inter_par_method'] = method
                var_opt['xE'] = xE
                var_opt['xU'] = xU
                var_opt['num_point'] = ind_exist
                var_opt['flag'] = flag
                var_opt['T_lorenz'] = T_lorenz
                var_opt['h_lorenz'] = h
                io.savemat("allpoints/pre_opt", var_opt)

                return
