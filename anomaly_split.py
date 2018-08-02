# -*- coding: utf-8 -*-
import math
import numpy as np
import statsmodels.tsa.stattools as ts
from math_ano import *

# this is an implement for anomaly detection (surus - netflix)
# original source code here: https://github.com/Netflix/Surus


#' @param X a vector representing a time series, or a data frame where columns are time series.
#' The length of this vector should be divisible by frequency.
#' If X is a vector it will be cast to a matrix of dimension frequency by length(X)/frequency
#' @param frequency the frequency of the seasonality of X
#' @param dates optional vector of dates to be used as a time index in the output
#' @param autodiff boolean. If true, use the Augmented Dickey Fuller Test to determine
#' if differencing is needed to make X stationary
#' @param isForceDiff boolean. If true, always compute differences
#' @param scale boolean. If true normalize the time series to zero mean and unit variance
#' @param lpenalty a scalar for the amount of thresholding in determining the low rank approximation for X.
#' The default values are chosen to correspond to the smart thresholding values described in Candes'
#' Stable Principal Component Pursuit
#' @param spenalty a scalar for the amount of thresholding in determining the separation between noise and sparse outliers
#' The default values are chosen to correspond to the smart thresholding values described in Zhou's
#' Stable Principal Component Pursuit
#' @param verbose boolean. If true print status updates while running optimization program

#' @return
# the outliers list (1: outlier point, 0: normal point)

def anomaly(X, frequency=7, dates=0, autodiff=True, isForceDiff=False,
            scale=True, lpenalty=1,
            spenalty=0,
            verbose=False):

    def computeL(mu):
        # S = np.zeros((7,9))
        LPenalty = lpenalty * mu
        X_temp = input2D - S
        U, Sing, V = np.linalg.svd(X_temp, full_matrices=False)

        penalizedD = softThreshold(Sing, LPenalty)
        D_matrix = np.zeros((len(penalizedD), len(penalizedD)))
        np.fill_diagonal(D_matrix, penalizedD)
        L = np.dot(np.dot(U, D_matrix), V)  ################# L la bien toan cuc
        return (sum(penalizedD) * LPenalty), L

    def computeS(mu):
        SPenalty = spenalty * mu
        X_temp = input2D - L
        penalizedS = softThreshold2(X_temp, SPenalty)
        S = penalizedS
        return ((l1norm(penalizedS.flatten()) * SPenalty), S)

    def computeE():
        E = input2D - L - S
        norm = np.linalg.norm(E)
        return (norm ** 2.0)

    def computeObjective(nuclearnorm, l1norm, l2norm):
        return 0.5 * l2norm + nuclearnorm + l1norm

    def computeDynamicMu():
        m = len(E)
        n = len(input2D[0])
        E_sd = sd(E.flatten())
        mu = E_sd * math.sqrt(2 * max(m, n));

        return max(.01, mu)

    # max loop in the While loop
    MAX_ITERS = 1000

    PVALUE_THRESHOLD = 0.05
    residual_len = len(X) % frequency

    X = X[residual_len:]

    nRows = frequency
    nCols = int(len(X) / frequency)
    minRecords = 2 * nRows

    # set default value for spenalty
    if spenalty == 0:
        spenalty = 1.4 / float(math.sqrt(max(nRows, nCols)))


    numNonZeroRecords = 0
    eps = 1e-12

    for i in range(len(X)):
        if (np.abs(X[i]) > eps):
            numNonZeroRecords = numNonZeroRecords + 1

    if (numNonZeroRecords >= minRecords):

        # call DickeyFullerTest
        lag_order = math.trunc((len(X) - 1.0) ** (1.0 / 3.0))  # default in R
        p_value = ts.adfuller(X, maxlag=lag_order, autolag=None, regression='ct')[1]
        dickey_needsDiff = False
        if (p_value > PVALUE_THRESHOLD):
            dickey_needsDiff = True

        zeroPaddedDiff = np.diff(X)
        zeroPaddedDiff = np.insert(zeroPaddedDiff, 0, 0.0)

        inputArrayTransformed = X

        if (autodiff == True and dickey_needsDiff == True):
            # Auto Diff
            inputArrayTransformed = zeroPaddedDiff
        elif isForceDiff == True:
            inputArrayTransformed = zeroPaddedDiff

        # calc mean:
        mean = np.mean(inputArrayTransformed)

        # calc sd:
        n = len(inputArrayTransformed)
        c = np.mean(inputArrayTransformed)
        ss = sum((x - c) ** 2 for x in inputArrayTransformed)
        pvar = ss / float(n - 1)  # the population variancepvar**0.5
        stdev = pvar ** 0.5

        inputArrayTransformed = np.asarray(inputArrayTransformed)
        inputArrayTransformed = inputArrayTransformed.astype(float)
        for i in range(0, len(inputArrayTransformed)):
            inputArrayTransformed[i] = (inputArrayTransformed[i] - mean) / float(stdev);

        input2D = VectorToMatrix(np.array(inputArrayTransformed), nRows, nCols)

        mu = nCols * nRows / float(4 * l1norm(inputArrayTransformed))

        objPrev = 0.5 * (np.linalg.norm(input2D) ** 2.0)

        obj = objPrev
        tol = 1e-8 * objPrev
        iter_ = 0
        diff = 2 * tol
        L = np.zeros((nRows, nCols))
        S = np.zeros((nRows, nCols))
        E = np.zeros((nRows, nCols))
        while (diff > tol and iter_ < MAX_ITERS):
            nuclearNorm, S = computeS(mu)
            l1Norm, L = computeL(mu)

            # calc E
            E = input2D - L - S
            norm = np.linalg.norm(E)
            l2Norm = (norm ** 2.0)

            obj = computeObjective(nuclearNorm, l1Norm, l2Norm)
            diff = abs(objPrev - obj)
            objPrev = obj
            mu = computeDynamicMu()
            iter_ = iter_ + 1

        outputS = S

        for i1 in range(nRows):
            for j1 in range(nCols):
                outputS[i1][j1] = outputS[i1][j1] * stdev
                #convert to 1 (outlier), 0 (normal):
                if ((outputS[i1][j1]) * (outputS[i1][j1]) < 0.0001):
                    outputS[i1][j1] = 0
                else:
                    outputS[i1][j1] = 1

        #convert array to list
        out = []
        for i in range(residual_len):
            out.append(0)
        r, c = np.shape(outputS)
        for i in range(0, c):
            for j in range(0, r):
                out.append(outputS[j][i])

        # return an list
        return out

# end the function
###### End file
