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


## plot final results in a bar chart

results = []
with open(f'results/final.csv') as f:
    lines = f.readlines()
    temp = []
    for line in lines:
        line = line.strip().split(',')
        line = [float(x) for x in line]
        results.append(line)
print(results)          
results = np.array(results).T
 
    
index = ['air_passengers', 'beer_production', 'electric_production', 'gold_price', 'yahoo_stock']
df = pd.DataFrame({'Elman RNN': results[0], 'Jordan RNN': results[1], 'Multi RNN': results[2]}, index=index)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Set the color palette (coolwarm, rocket)
# color_palette = sns.color_palette("rocket", n_colors=6)[::-1]
color_palette = sns.color_palette("Blues_d", n_colors=3)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_position([0.1, 0.1, 0.7, 0.8])  # Shift the bar chart to the left

df.plot.bar(ax=ax, rot=0, color=color_palette, zorder=2)

# Add horizontal grid lines behind the bars
ax.grid(axis='y', linestyle='--', zorder=1, alpha=0.7)

# Move the legend outside the plot
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)

ax.set_ylabel('MAPE (%)')
ax.set_xlabel('Dataset')
ax.set_title('Performance of RNN Architectures', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/results.png', bbox_inches='tight')
