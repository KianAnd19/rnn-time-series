import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from utils import polynomial_detrend 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

def adf_test(timeseries):
    result = adfuller(timeseries)
    
    if result[1] <= 0.05:
        return True
    return False

def kpss_test(timeseries):
    result = kpss(timeseries)
     
    if result[1] <= 0.05:
        return False
    return True

datasets = ['air_passengers', 'electric_production', 'minimum_temp', 'beer_production', 'gold_price', 'yahoo_stock']

for dataset in datasets:
    print('\n\nDataset: ', dataset)
    results = []
    with open(f'new_datasets/{dataset}.csv', 'r') as f:
        lines = f.readlines()
                
        for line in lines:
            line = line.split(',')
            date = datetime.strptime(line[0], '%d/%m/%Y')
            results.append([date, float(line[1])])

    results = np.array(results)
    adf_result = adf_test(results[:, 1])
    kpss_result = kpss_test(results[:, 1])
    print('ADF', adf_result, 'kpss', kpss_result)
    
    plt.figure(figsize=(10, 6))
    plt.tight_layout()
    plt.plot(results[:, 0], results[:, 1])
    
    if(not kpss_result and not adf_result):
        detrend = results.copy()
        detrend[:, 1] = polynomial_detrend(results[:, 1], 3)[0]
        print('ADF', adf_test(detrend[:, 1]), 'kpss', kpss_test(detrend[:, 1]))
        plt.plot(detrend[:, 0], detrend[:, 1])

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['Original Time Series', 'Detrended Time Series'])
    plt.title(dataset)
    plt.xticks(rotation=45)
    plt.savefig(f'figures/{dataset}.pdf')
