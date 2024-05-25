import numpy as np

# Activation function -> f()
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# and its derivative -> f'()
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# just for clarification that it's not a mistake
def linear(x):
    return x

class NNPIDAutotuner:
    input_size = 6  # Number of input neurons
    hidden_size = 6  # Number of hidden neurons
    output_size = 3  # Number of output neurons (Kp, Ki, Kd)

    def __init__(self, momentumA, momentumB, learning_rate):
        self.momentumA = momentumA
        self.momentumB = momentumB
        self.learning_rate = learning_rate
        # Initialize weights and biases
        self.W_input_hidden = np.random.rand(6, 6)
        self.W_hidden_output = np.random.rand(3, 6)
        self.bias_hidden = np.random.rand(6)
        self.bias_output = np.random.rand(3)
        # for hidden layer
        self.prev_deltaW1 = np.zeros((6, 6))
        self.prevPrev_deltaW1 = np.zeros((6, 6))
        # for output layer
        self.prev_deltaW2 = np.zeros((3, 6))
        self.prevPrev_deltaW2 = np.zeros((3, 6))

    
    def predict(self, X):
        '''
        This function predicts value of Kp, Ki, Kd deltas

        Args:
            X: NN input vector - [u_n, u_np, y_n, y_np, Uc_n, Uc_np]

        Returns:
            delta_Kp
            delta_Ki
            delta_Kd

        '''

        ''' Feedforward '''
        # create vector out of the input data from the system (6x1)
        # calculate input for hidden layer (6x1 vector) W*X = h_in 
        self.hidden_input = np.dot(self.W_input_hidden, X) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        # final/output layer calculations
        final_input = np.dot(self.W_hidden_output, self.hidden_output) + self.bias_output
        delta_Kp, delta_Ki, delta_Kd = linear(final_input)

        return delta_Kp, delta_Ki, delta_Kd

    def train(self, E, dY_du):
        '''
        This function trains NN online

        Args:
            dY_du: plant Jacobian at moment 'n+1'
            E: vector with system error at time accrodingly n, n-1, n-2 (y - y_m, diffrence between system output and reference model output)

        '''

        ''' Backpropagation - online'''
        delta_W2 = np.zeros((3, 6)) # matrix with deltas for each weight
        du_dok = np.array([E[0]-E[1], E[0], E[0]-2*E[1]-E[2]]) # look equation (14)
        # iterate over each element in output weights array (W_kj)
        for k in range(delta_W2.shape[0]):
            gradient_k = 0
            for j in range(delta_W2.shape[1]):
                delta_W2[j][i] = (self.learning_rate * gradient_k * self.hidden_output[j] +
                                  self.momentumA * self.prev_deltaW2[k][j] + 
                                  self.momentumB * self.prevPrev_deltaW2[k][j])
        # update prev deltas for W2
        self.prevPrev_deltaW2 = self.prev_deltaW2
        self.prev_deltaW2 = delta_W2
        # update weight matrix for hidden layer
        self.W_hidden_output += delta_W2
        
        # iterate over each element in hidden weights array
        delta_W1 = np.zeros((6, 6))
        for j in range(delta_W1.shape[0]):
            gradient_j = self.gradient_j(j)
            for i in range(delta_W1.shape[1]):
                delta_W1[j][i] = (self.learning_rate * gradient_j * self.hidden_input[i] +
                                  self.momentumA * self.prev_deltaW1[j][i] + 
                                  self.momentumB * self.prevPrev_deltaW1[j][i])
        # update prev deltas for W1
        self.prevPrev_deltaW1 = self.prev_deltaW1
        self.prev_deltaW1 = delta_W1
        # update weight matrix for hidden layer
        self.W_input_hidden += delta_W1

    def costFunc(e):
        return 0.5*e**2

