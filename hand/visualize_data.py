import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from utils import linear_detrend
from statsmodels.tsa.seasonal import seasonal_decompose

datasets = ['air_passengers', 'electric_production', 'minimum_temp', 'beer_production', 'gold_price', 'yahoo_stock']

def seasonal_detrend(data, period):
    decomposition = seasonal_decompose(data, period=period)
    detrended = data - decomposition.trend
    return detrended

for dataset in datasets:
    results = []
    with open(f'new_datasets/{dataset}.csv', 'r') as f:
        lines = f.readlines()
                
        for line in lines:
            line = line.split(',')
            date = datetime.strptime(line[0], '%d/%m/%Y')
            results.append([date, float(line[1])])

    results = np.array(results)

    detrend = results.copy()
    detrend[:, 1] = seasonal_detrend(results[:, 1], 12)

    plt.figure(figsize=(10, 6))
    plt.tight_layout()
    plt.plot(results[:, 0], results[:, 1])
    plt.plot(detrend[:, 0], detrend[:, 1])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(dataset)
    plt.xticks(rotation=45)
    plt.savefig(f'figures/{dataset}.pdf')
