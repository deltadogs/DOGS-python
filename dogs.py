import numpy as np
import scipy
import pandas as pd
from scipy import optimize
from scipy.spatial import Delaunay
import scipy.io as io
import os, inspect
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
Contributors: Shahrouz Ryan Alimo, Muhan Zhao
Modified: Oct. 2017 '''


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
            plt.title(r"$\Delta$-DOGS: " + sc + ': K = ' + str(Kini) + ' ' + str(len(p_iter)) + "th Iteration: RD = " + str(p) + ', MS = ' + str(Nm), y=1.05)
        elif sc == "AdaptiveK":
            plt.title(r"$\Delta$-DOGS: " + sc + ' ' + str(len(p_iter)) + "th Iteration: RD = " + str(p) + ', MS = ' + str(Nm), y=1.05)
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
            plt.title(r"$\Delta$-DOGS " + sc + ': K = ' + str(Kini) + ' ' + str(len(p_iter)) + "th Iteration: " + 'MeshSize = ' + str(Nm), y=1.05)
        elif sc == "AdaptiveK":
            plt.title(r"$\Delta$-DOGS: " + sc + ' ' + str(len(p_iter)) + "th Iteration: " + 'MeshSize = ' + str(Nm), y=1.05)

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
        plot_folder = current_path + "/plot" + '/' + str(ff)
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
    # Save fig
    plt.savefig('plot/' + str(ff) + '/DR' + str(len(p_iter)) +'.png', format='png', dpi=1000)
    return



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
