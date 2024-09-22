import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from elman import ElmanRNN
from jordan import JordanRNN
from multi import MultiRNN

def preprocess_data(filename, sequence_length=10):
    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        data = []
        for line in lines:
            date_str, production = line.strip().split(",")
            date = datetime.strptime(date_str, "%Y-%m-%d")
            data.append([date.timestamp(), float(production)])
    
    data = np.array(data)
    
    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data[:, 1].reshape(-1, 1))
    
    # Create sequences
    X, Y = [], []
    for i in range(len(normalized_data) - sequence_length):
        X.append(normalized_data[i:i+sequence_length])
        Y.append(normalized_data[i+sequence_length])
    
    return np.array(X), np.array(Y), scaler

# Set up the RNN
input_size = 5  # sequence length
hidden_size = 20
output_size = 1
learning_rate = 1e-4

rnn = ElmanRNN(input_size, hidden_size, output_size, learning_rate=learning_rate)

# Preprocess the data
# X, Y, scaler = preprocess_data("datasets/Electric_Production.csv", sequence_length=input_size)
X, Y, scaler = preprocess_data("datasets/gold_price_data.csv", sequence_length=input_size)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# Train the model
rnn.train(X_train, Y_train, epochs=1000)

# Make predictions
predictions = rnn.predict(X_test)

# Inverse transform the predictions and actual values
predictions_original = scaler.inverse_transform(predictions)
Y_test_original = scaler.inverse_transform(Y_test)

# Print some results
print("Predictions:", predictions_original[:5].flatten())
print("Actual:", Y_test_original[:5].flatten())

# Calculate Mean Absolute Error
mae = np.mean(np.abs(predictions_original - Y_test_original))
print(f"Mean Absolute Error: {mae}")