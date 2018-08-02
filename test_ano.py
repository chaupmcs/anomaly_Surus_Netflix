# -*- coding: utf-8 -*-
#### testing for anomaly function_Sep-1st-2016
import numpy as np
import pandas as pd
from anomaly_split import anomaly #import anomaly function for testing
import time


# the test function

### param:
####### data: time series (multi-columns, each col is a testcase)
####### result_R: The output from original function (R package) (multi-cols)
####### n: the number of testcases (max(n) = number of cols(data)

### return:  fail, pass statuses
def test_ano(data, result_R, n):

   #count pass, fail:
    failed = 0
    passed = 0

    print("testing... please wait")
    for i in range(n):
        x = data[data.columns[(1+i)]]#because the first col is Date, so we get it away by plus 1 (i.e: i+1)
        X = pd.Series.tolist(x) # convert series type to list

        # the result we wanna test
        res_py =np.asarray(anomaly(X)) #convert to array

        #read the "true" result (expected values)
        res_r = result_R[result_R.columns[(i)]]
        res_r = pd.Series.tolist(res_r) # convert series type to list

        #complete missing values in R:
        ### perhap the length(result_R) is different from length(result_python).
        ### That why we need these stuff:
        sub_len = len(res_py) - len(res_r)
        for j in range(sub_len):
            res_r = [0]+res_r # fill "0"(s) at the begin of res_r, until the both length is the same

        # convert res_r to array
        res_r = np.asarray(res_r)
        
        #check different:
        ans=abs(res_r-res_py)
        
        if (sum(ans) < 1):
            print("testcase ", (i+1), " passed")
            passed+=1
        else: #if It has any diffrences (failed)
            print("testcase ", (i+1), " failed !!!!!!!!!!!!!!!!!!")
            failed+=1

    print("===== Summary: ======")
    print("passed: ", passed)
    print("failed: ", failed)
#end 'test_ano' function






def run_test():

    # read file csv (the first col is DATE, others are timeseries)
    data = pd.read_csv("C:\\Users\\CPU10902-local\\Desktop\\op_msg.csv", na_values='NA')

    # The expected values (true result)
    result_R = pd.read_csv("C:\\Users\\CPU10902-local\\Desktop\\resut_R.csv", na_values='NA')

    # call the test function
    test_ano(data, result_R, 42)
#end 'run_test' function

if __name__ == '__main__':
    start_time = time.time()
    run_test()
    print("--- total run time: %s seconds ---" % (time.time() - start_time))
#### result:
    '''
    testing... please wait
    ('testcase ', 1, ' passed')
    ('testcase ', 2, ' passed')
    ('testcase ', 3, ' passed')
    ('testcase ', 4, ' passed')
    ('testcase ', 5, ' passed')
    ('testcase ', 6, ' passed')
    ('testcase ', 7, ' passed')
    ('testcase ', 8, ' passed')
    ('testcase ', 9, ' passed')
    ('testcase ', 10, ' passed')
    ('testcase ', 11, ' passed')
    ('testcase ', 12, ' passed')
    ('testcase ', 13, ' passed')
    ('testcase ', 14, ' passed')
    ('testcase ', 15, ' passed')
    ('testcase ', 16, ' passed')
    ('testcase ', 17, ' passed')
    ('testcase ', 18, ' passed')
    ('testcase ', 19, ' passed')
    ('testcase ', 20, ' passed')
    ('testcase ', 21, ' passed')
    ('testcase ', 22, ' passed')
    ('testcase ', 23, ' passed')
    ('testcase ', 24, ' passed')
    ('testcase ', 25, ' passed')
    ('testcase ', 26, ' passed')
    ('testcase ', 27, ' passed')
    ('testcase ', 28, ' passed')
    ('testcase ', 29, ' passed')
    ('testcase ', 30, ' passed')
    ('testcase ', 31, ' passed')
    ('testcase ', 32, ' passed')
    ('testcase ', 33, ' passed')
    ('testcase ', 34, ' passed')
    ('testcase ', 35, ' passed')
    ('testcase ', 36, ' passed')
    ('testcase ', 37, ' passed')
    ('testcase ', 38, ' passed')
    ('testcase ', 39, ' passed')
    ('testcase ', 40, ' passed')
    ('testcase ', 41, ' passed')
    ('testcase ', 42, ' passed')
    ===== Summary: ======
    ('passed: ', 42)
    ('failed: ', 0)
--- total run time: 10.2110002041 seconds ---
    '''
### end file "test_ano" ###