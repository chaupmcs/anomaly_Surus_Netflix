import numpy as np
import pandas as pd
from anomaly_split import anomaly #import anomaly function for testing
import time

def run_test():
    # read file csv (the first col is DATE, others are timeseries)
    data = pd.read_csv("C:\\Users\\CPU10902-local\\Desktop\\data_test\\data_hourly_cut_60_dates.csv", na_values='NA')
    result_table = data

    n = len(data.columns)
    ano_point = [None] * (n-1)
    print("running... please wait")
    for i in range((n-1)):
        x = data[data.columns[(1 + i)]]  # because the first col is Date, so we get it away by plus 1 (i.e: i+1)
        X = pd.Series.tolist(x)  # convert series type to list

        # the result we wanna test
        ano_point[i] = np.asarray(anomaly(X=X, frequency = 24))  # convert to array

    #print 'ano_point', ano_point
    ano_point = pd.DataFrame(ano_point).transpose()




    ano_point.columns = result_table.columns[1:] + '_anotrend'

    result_table = pd.concat([result_table.reset_index(drop=True), ano_point], axis=1)
    result_table.to_csv("C:\\Users\\CPU10902-local\\Desktop\\ano_point_old_package.csv", sep=',', index=False)

    return ano_point


# end 'run_test' function

if __name__ == '__main__':
    start_time = time.time()
    run_test()
    print("--- Done.  Total run time: %s seconds ---" % (time.time() - start_time))