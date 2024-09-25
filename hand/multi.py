import numpy as np
from utils import mse_loss, mse_loss_derivative, Adam

class MultiRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W_ih = np.random.randn(hidden_size, input_size) * 0.01
        self.W_ho = np.random.randn(output_size, hidden_size) * 0.01
        self.W_oy = np.random.randn(hidden_size, output_size + hidden_size) * 0.01  # Feedback connection
        
        # Initialize biases
        self.b_h = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((output_size, 1))

        # Initialize Adam optimizers
        self.optimizers = {
            'W_ih': Adam(learning_rate),
            'W_ho': Adam(learning_rate),
            'W_oy': Adam(learning_rate),
            'b_h': Adam(learning_rate),
            'b_o': Adam(learning_rate)
        }
        
    def forward(self, x, h_prev, y_prev):
        # Concatenate previous output and previous hidden state
        prev_state = np.concatenate((y_prev, h_prev), axis=0)
        
        # Input to hidden (including feedback from previous output and hidden state)
        h = np.tanh(np.dot(self.W_ih, x) + np.dot(self.W_oy, prev_state) + self.b_h)
        
        # Hidden to output
        y = np.dot(self.W_ho, h) + self.b_o
        
        return y, h
    
    def backward(self, x, y_prev, h_prev, h, y_pred, y_true):
        dL_dy = mse_loss_derivative(y_true, y_pred)
        dL_dWho = np.dot(dL_dy, h.T)
        dL_dbo = dL_dy
        
        dL_dh = np.dot(self.W_ho.T, dL_dy)
        dL_dh_raw = dL_dh * (1 - h**2)
        
        prev_state = np.concatenate((y_prev, h_prev), axis=0)
        dL_dWih = np.dot(dL_dh_raw, x.T)
        dL_dWoy = np.dot(dL_dh_raw, prev_state.T)
        dL_dbh = dL_dh_raw
    
        return dL_dWih, dL_dWoy, dL_dbh, dL_dWho, dL_dbo
    
    def update_parameters(self, dL_dWih, dL_dWoh, dL_dbh, dL_dWho, dL_dbo):
        self.W_ih = self.optimizers['W_ih'].update(self.W_ih, dL_dWih)
        self.W_ho = self.optimizers['W_ho'].update(self.W_ho, dL_dWho)
        self.W_oy = self.optimizers['W_oy'].update(self.W_oy, dL_dWoh)
        self.b_h = self.optimizers['b_h'].update(self.b_h, dL_dbh)
        self.b_o = self.optimizers['b_o'].update(self.b_o, dL_dbo)
    
    def train(self, X, Y, epochs, verbose):
        for epoch in range(epochs):
            y_prev = np.zeros((self.output_size, 1))
            h_prev = np.zeros((self.hidden_size, 1))
            total_loss = 0
            
            for t in range(len(X)):
                x = X[t].reshape(-1, 1)
                y_true = Y[t].reshape(-1, 1)
                
                # Forward pass
                y_pred, h_next = self.forward(x, h_prev, y_prev)
                
                # Compute loss
                loss = mse_loss(y_true, y_pred)
                total_loss += loss
                
                # Backward pass
                gradients = self.backward(x, y_prev, h_prev, h_next, y_pred, y_true)
                
                # Update parameters
                self.update_parameters(*gradients)
                
                y_prev = y_pred
            
            if epoch % 100 == 0 and verbose:
                print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")
    
    
    def predict(self, X):
        y_prev = np.zeros((self.output_size, 1))
        h_prev = np.zeros((self.hidden_size, 1))
        predictions = []
        
        for x in X:
            x = x.reshape(-1, 1)
            y, h = self.forward(x, h_prev, y_prev)
            predictions.append(y.flatten())
            y_prev = y
            h_prev = h
        
        return np.array(predictions)
