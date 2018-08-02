import numpy as np
#help functions for anomaly function
# calc standard deviation
def sd(data):  
    n = len(data)
    c = np.mean(data)
    ss = sum((x-c)**2 for x in data)
    pvar = ss/(n-1) # the population variance
    return pvar**0.5

#convert vector to matrix
def VectorToMatrix(x, rows, cols):
        return x.reshape(rows,cols, order='F')
        
   


def softThreshold(x, penalty):
    res = []
    for i in range(len(x)):
        if x[i] > 0:
            res += [max(abs(x[i]) - penalty, 0)]
        elif x[i] < 0:
            res += [- (max(abs(x[i]) - penalty, 0))]
        else:
            res += [0]
    return res

    
def softThreshold2(X, penalty):
    numrowsX = len(X)    
    numcolsX = len(X[0])
    res = np.zeros((numrowsX, numcolsX))
    
    for i in range(numrowsX):
        for j in range(numcolsX):
            if X[i][j] > 0:
                res[i][j] = max(abs(X[i][j]) - penalty, 0)
            elif X[i][j] < 0:
                res[i][j] = - (max(abs(X[i][j]) - penalty, 0))
            else:
                res[i][j]  = 0
    return res


# convert the data x to abs(x), then calc sum
def l1norm(x):
    l1norm = 0
    for i in range(0, len(x)):
        l1norm += np.abs(x[i])
    return l1norm
    
