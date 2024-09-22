import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from network import ElmanRNN, JordanRNN, MultiLayerRNN, train_rnn, predict_rnn

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate some dummy data
X = np.random.randn(1000, 2).astype(np.float32)
Y = np.sum(X, axis=1, keepdims=True).astype(np.float32)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
Y_tensor = torch.FloatTensor(Y)

# Create dataset and dataloader
dataset = TensorDataset(X_tensor, Y_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define network parameters
input_size = 2
hidden_size = 10
output_size = 1
hidden_sizes = [10, 8, 6]  # For Multi-Layer RNN

# Create instances of RNNs
elman_rnn = ElmanRNN(input_size, hidden_size, output_size)
jordan_rnn = JordanRNN(input_size, hidden_size, output_size)
multi_rnn = MultiLayerRNN(input_size, hidden_sizes, output_size)

# Define loss function and optimizers
criterion = nn.MSELoss()
elman_optimizer = optim.Adam(elman_rnn.parameters(), lr=0.001)
jordan_optimizer = optim.Adam(jordan_rnn.parameters(), lr=0.001)
multi_optimizer = optim.Adam(multi_rnn.parameters(), lr=0.001)

# Train the models
print("Training Elman RNN:")
train_rnn(elman_rnn, criterion, elman_optimizer, data_loader, num_epochs=1000)

print("\nTraining Jordan RNN:")
train_rnn(jordan_rnn, criterion, jordan_optimizer, data_loader, num_epochs=1000)

print("\nTraining Multi-Layer RNN:")
train_rnn(multi_rnn, criterion, multi_optimizer, data_loader, num_epochs=1000)

# Make predictions
test_data = torch.FloatTensor(np.random.randn(5, 2).astype(np.float32))
print("\nElman RNN Predictions:", predict_rnn(elman_rnn, test_data))
print("Jordan RNN Predictions:", predict_rnn(jordan_rnn, test_data))
print("Multi-Layer RNN Predictions:", predict_rnn(multi_rnn, test_data))
print("Actual:", np.sum(test_data.numpy(), axis=1))