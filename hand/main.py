import numpy as np

from elman import ElmanRNN
from jordan import JordanRNN
from multi import MultiRNN

# Example usage
input_size = 2
hidden_size = 5
output_size = 1

rnn = MultiRNN(input_size, hidden_size, output_size)

# Generate some dummy data
X = np.random.randn(100, input_size)
Y = np.sum(X, axis=1, keepdims=True)

# Train the model
rnn.train(X, Y, epochs=1000)

# Make predictions
predictions = rnn.predict(X)
print("Predictions:", predictions[:5])
print("Actual:", Y[:5])