import torch
import torch.nn as nn

class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ElmanRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = torch.tanh(self.i2h(combined))
        output = self.h2o(hidden)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

class JordanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(JordanRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Linear(input_size + output_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, prev_output):
        combined = torch.cat((input, prev_output), 1)
        hidden = torch.tanh(self.i2h(combined))
        output = self.h2o(hidden)
        return output
    
    def init_output(self, batch_size):
        return torch.zeros(batch_size, self.output_size)

class MultiLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MultiLayerRNN, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.rnn = nn.RNN(input_size, hidden_sizes[-1], num_layers=len(hidden_sizes), nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.fc(output)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(len(self.hidden_sizes), batch_size, self.hidden_sizes[-1])

# Training function
def train_rnn(model, criterion, optimizer, data_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            if isinstance(model, ElmanRNN):
                hidden = model.init_hidden(data.size(0))
                output, _ = model(data, hidden)
            elif isinstance(model, JordanRNN):
                prev_output = model.init_output(data.size(0))
                output = model(data, prev_output)
            elif isinstance(model, MultiLayerRNN):
                hidden = model.init_hidden(data.size(0))
                output, _ = model(data.unsqueeze(1), hidden)
                output = output.squeeze(1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss / len(data_loader)}')

# Prediction function
def predict_rnn(model, data):
    model.eval()
    with torch.no_grad():
        if isinstance(model, ElmanRNN):
            hidden = model.init_hidden(data.size(0))
            output, _ = model(data, hidden)
        elif isinstance(model, JordanRNN):
            prev_output = model.init_output(data.size(0))
            output = model(data, prev_output)
        elif isinstance(model, MultiLayerRNN):
            hidden = model.init_hidden(data.size(0))
            output, _ = model(data.unsqueeze(1), hidden)
            output = output.squeeze(1)
    return output