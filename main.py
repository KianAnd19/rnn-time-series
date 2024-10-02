import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from elman import ElmanRNN
from jordan import JordanRNN
from multi import MultiRNN
from utils import polynomial_detrend
import itertools
from tqdm import tqdm

rnns = [ElmanRNN, JordanRNN, MultiRNN]
names = ['Elman', 'Jordan', 'Multi']
datasets = ['air_passengers', 'electric_production', 'beer_production', 'gold_price', 'yahoo_stock']

def preprocess_data(filename, sequence_length=10, count=0):
    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        data = []
        for line in lines:
            date_str, production = line.strip().split(",")
            date = datetime.strptime(date_str, "%d/%m/%Y")
            data.append([date.timestamp(), float(production)])

    trend = None
    data = np.array(data)
    data[:, 1], trend = polynomial_detrend(data[:, 1], 3)

    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data[:, 1].reshape(-1, 1))
    
    # Create sequences
    X, Y = [], []
    for i in range(len(normalized_data) - sequence_length):
        X.append(normalized_data[i:i+sequence_length])
        Y.append(normalized_data[i+sequence_length])
    
    return np.array(X), np.array(Y), scaler, trend

# train on supplied dataset, then test and return accuracy.
def train_test(rnn, X_train, X_test, Y_train, Y_test, epochs, scaler, trend=None):
    rnn.train(X_train, Y_train, epochs=epochs)
    predictions = rnn.predict(X_test)

    # Make predictions
    predictions = rnn.predict(X_test)

    # Inverse transform the predictions and actual values
    predictions_original = scaler.inverse_transform(predictions)
    Y_test_original = scaler.inverse_transform(Y_test)

    if trend is not None:
        # Ensure trend is the correct length
        trend_subset = trend[-len(predictions_original):]
        predictions_original = predictions_original.flatten() + trend_subset
        Y_test_original = Y_test_original.flatten() + trend_subset
    

    # Calculate Mean Absolute Error
    mape = np.mean(np.abs(Y_test_original-predictions_original)/Y_test_original)*100
    mae = np.mean(np.abs(Y_test_original-predictions_original))
    mse = np.mean(np.sqrt(np.abs(Y_test_original-predictions_original)))
    rmse = np.sqrt(np.mean(np.power(Y_test_original-predictions_original, 2))) 
    
    return mae, rmse, mape, mse

def cv_validation(rnn, X, Y, scaler, trend, epochs):
    size_split = int((len(X) / k)*0.8)
    
    avg_mae = 0
    avg_rsme = 0
    avg_mape = 0
    avg_mse = 0

    for i in range(k-1, -1, -1):
        total_sample = round(1 - ((1/k)*i), 2)
        split = int(0.8 * total_sample * len(X))
        total_sample = int((total_sample) * len(X))

        X_train, X_test = X[split-size_split:split], X[split:total_sample]
        Y_train, Y_test = Y[split-size_split:split], Y[split:total_sample]
    
        mae, rsme, mape, mse = train_test(rnn, X_train, X_test, Y_train, Y_test, epochs, scaler, trend)
        avg_mae += mae
        avg_rsme += rsme
        avg_mape += mape
        avg_mse += mse
    avg_mae /= k
    avg_rsme /= k
    avg_mape /= k
    avg_mse /= k
    return avg_mae, avg_rsme, avg_mape, avg_mse

def grid_search():
    results = [[], [], []]

    hyperparameters = {
        'input_size': [5, 10, 15],
        'hidden_size': [5, 10, 15],
        'lr': [1e-2, 1e-3, 1e-4],
        'epochs': [500, 1000, 1500]
    }

    # list of all the combinations of hyperparameters
    param_grid = list(itertools.product(*hyperparameters.values()))

    for param in tqdm(param_grid):
        count = 0
        for rnn in rnns:
            temp = []
            for i in range(len(datasets)):
                input_size = param[0]
                hidden_size = param[1]
                output_size = 1
                learning_rate = param[2]
                
                X, Y, scaler, trend = preprocess_data(f'new_datasets/{datasets[i]}.csv', sequence_length=input_size, count=i)
        
                split = int(0.8*len(X))
        
                X_train, X_test = X[:split], X[split:]
                Y_train, Y_test = Y[:split], Y[split:]
        
                r = rnn(input_size, hidden_size, output_size, learning_rate=learning_rate)
                mae, rsme, mape, mse = cv_validation(r, X_train, Y_train, scaler, trend, epochs)
                temp.append(str(mae))
            
            results[count].append(temp) 
            count += 1

    ## print results to results/
    for i in range(len(names)):
        with open(f'results/{names[i]}.csv', 'w') as f:
            for j in range(len(results[i])):
                f.write(f'{param_grid[j][0]},{param_grid[j][1]},{param_grid[j][2]},{param_grid[j][3]},' + ','.join(results[i][j])+'\n')

def final_runs():
    for rnn, name in zip(rnns, names):
        print(f'{name}:')
        with open(f'results/{name}.csv', 'r') as f:
            results = []
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                line = [float(x) for x in line]
                results.append(line)

            for dataset, i in zip(datasets, range(len(datasets))):
                best = [0, np.inf]
                for j in range(len(results)):
                    if(results[j][i+4] < best[1]):
                       best = [j, results[j][i+4]]

                hyperparameters = results[best[0]][:4]

                # setting the input, hidden and epochs back to int
                hyperparameters[0] = int(hyperparameters[0])
                hyperparameters[1] = int(hyperparameters[1])
                hyperparameters[3] = int(hyperparameters[3])

                print(dataset, ': ', best[1], hyperparameters)
                
                X, Y, scaler, trend = preprocess_data(f'new_datasets/{datasets[i]}.csv', sequence_length=hyperparameters[0], count=i)

                
                split = int(0.8*len(X))
        
                X_train, X_test = X[:split], X[split:]
                Y_train, Y_test = Y[:split], Y[split:]
                
                r = rnn(hyperparameters[0], hyperparameters[1], 1, learning_rate=hyperparameters[2])

                mae, rsme, mape, mse = train_test(r, X_train, X_test, Y_train, Y_test, epochs, scaler, trend)
                
                print(f'MSE: {mse}, MAE: {mae}, MAPE: {mape}, RSME: {rsme}\n')


######################################################
#################### main ############################
######################################################

# Set up the RNN
#input_size = 15  # sequence length
#hidden_size = 10
#output_size = 1
#learning_rate = 1e-3
k = 5 # number of folds for blocked time series split 
epochs = 1000

#grid_search()
final_runs()
