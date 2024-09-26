import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from elman import ElmanRNN
from jordan import JordanRNN
from multi import MultiRNN
from utils import linear_detrend

rnns = [ElmanRNN, JordanRNN, MultiRNN]
datasets = ['air_passengers', 'electric_production', 'minimum_temp', 'beer_production', 'gold_price', 'yahoo_stock']

def preprocess_data(filename, sequence_length=10):
    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        data = []
        for line in lines:
            date_str, production = line.strip().split(",")
            date = datetime.strptime(date_str, "%d/%m/%Y")
            data.append([date.timestamp(), float(production)])
    
    data = np.array(data)
    # data[:, 1] = linear_detrend(data[:, 1])

    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data[:, 1].reshape(-1, 1))
    
    # Create sequences
    X, Y = [], []
    for i in range(len(normalized_data) - sequence_length):
        X.append(normalized_data[i:i+sequence_length])
        Y.append(normalized_data[i+sequence_length])
    
    return np.array(X), np.array(Y), scaler

# train on supplied dataset, then test and return accuracy.
def train_test(rnn, X_train, X_test, Y_train, Y_test, epochs):
    rnn.train(X_train, Y_train, epochs=epochs)
    predictions = rnn.predict(X_test)

    # Make predictions
    predictions = rnn.predict(X_test)

    # Inverse transform the predictions and actual values
    predictions_original = scaler.inverse_transform(predictions)
    Y_test_original = scaler.inverse_transform(Y_test)

    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(Y_test_original-predictions_original))
    
    return mae

######################################################
#################### main ############################
######################################################

# Set up the RNN
input_size = 15  # sequence length
hidden_size = 10
output_size = 1
learning_rate = 1e-4
k = 5 # number of folds for k-fold cross validation
epochs = 1000

for rnn in rnns:
    for i in range(5):
        print('Dataset: ', datasets[i])
        # Preprocess the data
        X, Y, scaler = preprocess_data(f'new_datasets/{datasets[i]}.csv', sequence_length=input_size)
        
        size_split = int((len(X) / k)*0.8)
        
        avg_result = 0

        for i in range(k-1, -1, -1):
            rnn = rnns[0](input_size, hidden_size, output_size, learning_rate=learning_rate)
            total_sample = round(1 - ((1/k)*i), 2)
            split = int(0.8 * total_sample * len(X))
            total_sample = int(total_sample * len(X))
        
            X_train, X_test = X[split-size_split:split], X[split:total_sample]
            Y_train, Y_test = Y[split-size_split:split], Y[split:total_sample]
        
            result = train_test(rnn, X_train, X_test, Y_train, Y_test, epochs)
            avg_result += result 

            print(total_sample, '\t', result)

        avg_result /= k
        print('Average Result: ', avg_result)
    break
