import numpy as np
from utils import linear_detrend

new = []
#d/m/y

with open('datasets/AirPassengers.csv', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split(',')
        date = line[0].split('-')
        new_date = f'01/{date[1]}/{date[0]}'
        new.append([new_date, float(line[1])])


new = np.array(new)
#new[:, 1] = linear_detrend(new[:, 1])

with open(f'new_datasets/air_passengers.csv', 'w') as f:
    for line in new:
        f.write(f'{line[0]},{line[1]}\n')
