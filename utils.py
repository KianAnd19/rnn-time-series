import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def xavier_distribution(input_size, output_size):
    temp = (6 ** 0.5) / ((input_size + output_size) ** 0.5)
    return np.random.uniform(-temp, temp, (input_size, output_size))

def linear_detrend(data):
    x = np.arange(len(data)).astype('float64')
    y = data.astype('float64')
    slope, intercept = np.polyfit(x, y, 1)
    trend = x * slope + intercept
    detrended = y - trend
    return detrended

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad_w):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grad_w)
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

