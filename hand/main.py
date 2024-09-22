import numpy as np

from elman import ElmanRNN
from jordan import JordanRNN
from multi import MultiRNN

# Example usage
input_size = 2
hidden_size = 10
output_size = 1

rnn = ElmanRNN(input_size, hidden_size, output_size, learning_rate=0.001)

# Generate some dummy data
X = np.random.randn(1000, 2).astype(np.float32)
Y = np.sum(X, axis=1, keepdims=True).astype(np.float32)

# Train the model
rnn.train(X, Y, epochs=1000)

# Make predictions
predictions = rnn.predict(X)
print("Predictions:", predictions[:5])
print("Actual:", Y[:5])