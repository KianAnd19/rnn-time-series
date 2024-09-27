import numpy as np
from utils import mse_loss, mse_loss_derivative, xavier_distribution, Adam

class JordanRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W_ih = xavier_distribution(hidden_size, input_size)
        self.W_ho = xavier_distribution(output_size, hidden_size)
        self.W_oy = xavier_distribution(hidden_size, output_size)  # Feedback connection
        
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
        
    def forward(self, X, y_init):
        T = len(X)
        h = np.zeros((T + 1, self.hidden_size, 1))
        y = np.zeros((T + 1, self.output_size, 1))
        y[0] = y_init
        
        for t in range(T):
            x = X[t].reshape(-1, 1)
            h[t+1] = np.tanh(np.dot(self.W_ih, x) + np.dot(self.W_oy, y[t]) + self.b_h)
            y[t+1] = np.dot(self.W_ho, h[t+1]) + self.b_o
        
        return y[1:], h

    def backward(self, X, Y, h, y_pred):
        T = len(X)
        dL_dWih = np.zeros_like(self.W_ih)
        dL_dWoy = np.zeros_like(self.W_oy)
        dL_dbh = np.zeros_like(self.b_h)
        dL_dWho = np.zeros_like(self.W_ho)
        dL_dbo = np.zeros_like(self.b_o)
        
        dL_dh_next = np.zeros_like(h[0])
        dL_dy_prev = np.zeros((self.output_size, 1))
        
        for t in reversed(range(T)):
            dL_dy = mse_loss_derivative(Y[t].reshape(-1, 1), y_pred[t]) + dL_dy_prev
            
            dL_dWho += np.dot(dL_dy, h[t+1].T)
            dL_dbo += dL_dy
            
            dL_dh = np.dot(self.W_ho.T, dL_dy) + dL_dh_next
            dL_dh_raw = dL_dh * (1 - h[t+1]**2)
            
            dL_dWih += np.dot(dL_dh_raw, X[t].reshape(1, -1))
            if t > 0:
                dL_dWoy += np.dot(dL_dh_raw, y_pred[t-1].T)
            dL_dbh += dL_dh_raw
            
            dL_dy_prev = np.dot(self.W_oy.T, dL_dh_raw)
        
        return dL_dWih, dL_dWoy, dL_dbh, dL_dWho, dL_dbo

    
    def update_parameters(self, dL_dWih, dL_dWoh, dL_dbh, dL_dWho, dL_dbo):
        self.W_ih = self.optimizers['W_ih'].update(self.W_ih, dL_dWih)
        self.W_ho = self.optimizers['W_ho'].update(self.W_ho, dL_dWho)
        self.W_oy = self.optimizers['W_oy'].update(self.W_oy, dL_dWoh)
        self.b_h = self.optimizers['b_h'].update(self.b_h, dL_dbh)
        self.b_o = self.optimizers['b_o'].update(self.b_o, dL_dbo)
    
    def train(self, X, Y, epochs, verbose=False):
        for epoch in range(epochs):
            y_init = np.zeros((self.output_size, 1))
            total_loss = 0
            
            # Forward pass
            y_pred, h = self.forward(X, y_init)
            
            # Compute loss
            loss = sum([mse_loss(Y[t].reshape(-1, 1), y_pred[t]) for t in range(len(X))])
            total_loss += loss
            
            # Backward pass
            gradients = self.backward(X, Y, h, y_pred)
            
            # Update parameters
            self.update_parameters(*gradients)
            
            if epoch % 100 == 0 and verbose:
                print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")
    
    def predict(self, X):
        y_init = np.zeros((self.output_size, 1))
        y_pred, _ = self.forward(X, y_init)
        return np.array([y.flatten() for y in y_pred])
