import numpy as np

# Defining the activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Function to generate training data for logical functions
def generate_data(function, n_inputs):
    # Generate all possible combinations of boolean inputs
    inputs = np.array(np.meshgrid(*[[0, 1]] * n_inputs)).T.reshape(-1, n_inputs)

    # Calculate the corresponding output for the chosen logical function
    if function == 'AND':
        outputs = np.all(inputs, axis=1).astype(int)
    elif function == 'OR':
        outputs = np.any(inputs, axis=1).astype(int)
    elif function == 'XOR':
        outputs = np.bitwise_xor.reduce(inputs, axis=1)
    else:
        raise ValueError("Unknown function")

    return inputs, outputs.reshape(-1, 1)

# Defining the neural network
class NeuralNetwork:
    def __init__(self, n_inputs):
        # Initializing weights randomly
        self.weights1 = np.random.rand(n_inputs, 4)  # Weights of the hidden layer
        self.weights2 = np.random.rand(4, 1)  # Weights of the output layer

    def feedforward(self, inputs):
        self.layer1 = sigmoid(np.dot(inputs, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    def backpropagation(self, inputs, outputs, learning_rate):
        # Calculating the error
        error = outputs - self.feedforward(inputs)

        # Calculating the delta (adjustment) for the weights
        d_weights2 = np.dot(self.layer1.T, (2 * error * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(inputs.T, (
                np.dot(2 * error * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(
            self.layer1)))

        # Updating the weights
        self.weights1 += learning_rate * d_weights1
        self.weights2 += learning_rate * d_weights2

    def train(self, inputs, outputs, learning_rate=0.1, epochs=10000):
        for _ in range(epochs):
            self.backpropagation(inputs, outputs, learning_rate)

# Function to create and train the neural network
def train_network(function, n_inputs):
    inputs, outputs = generate_data(function, n_inputs)
    neural_network = NeuralNetwork(n_inputs)
    neural_network.train(inputs, outputs)
    return neural_network

class ImprovedNeuralNetwork:
    def __init__(self, n_inputs):
        # Increasing the number of neurons in the hidden layers
        self.weights1 = np.random.rand(n_inputs, 8)  # First hidden layer
        self.weights2 = np.random.rand(8, 8)  # Second hidden layer
        self.weights3 = np.random.rand(8, 1)  # Output layer

    def feedforward(self, inputs):
        self.layer1 = sigmoid(np.dot(inputs, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer2, self.weights3))
        return self.output

    def backpropagation(self, inputs, outputs, learning_rate):
        # Calculating the error
        error = outputs - self.feedforward(inputs)

        # Calculating the delta (adjustment) for the weights
        d_weights3 = np.dot(self.layer2.T, (2 * error * sigmoid_derivative(self.output)))
        d_weights2 = np.dot(self.layer1.T, (
                np.dot(2 * error * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(
            self.layer2)))
        d_weights1 = np.dot(inputs.T, (np.dot(
            np.dot(2 * error * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2),
            self.weights2.T) * sigmoid_derivative(self.layer1)))

        # Updating the weights
        self.weights1 += learning_rate * d_weights1
        self.weights2 += learning_rate * d_weights2
        self.weights3 += learning_rate * d_weights3

    def train(self, inputs, outputs, learning_rate=0.1, epochs=50000):
        for _ in range(epochs):
            self.backpropagation(inputs, outputs, learning_rate)

# Modified function to train the neural network
def train_improved_network(function, n_inputs):
    inputs, outputs = generate_data(function, n_inputs)
    neural_network = ImprovedNeuralNetwork(n_inputs)
    neural_network.train(inputs, outputs, learning_rate=0.1, epochs=50000)
    return neural_network

# Request the number of inputs and the desired function
num_inputs = int(input("Enter the number of inputs (2, 3, 4... n): "))
func_type = input("Enter the desired function (AND, OR, XOR): ").strip().upper()

# Check if the provided function is valid
if func_type not in ['AND', 'OR', 'XOR']:
    raise ValueError("Unknown function. Choose between AND, OR, and XOR.")

# Generate all possible "n" boolean inputs
boolean_inputs = generate_data(func_type, num_inputs)[0]

# Train the appropriate neural network based on the desired function
if func_type in ['AND', 'OR']:
    nn = train_network(func_type, num_inputs)
else:
    nn = train_improved_network(func_type, num_inputs)

# Test the neural network with the generated inputs
results = nn.feedforward(boolean_inputs)
print(f"INPUTS:\n{boolean_inputs}")
print(f"{func_type} RESULTS:\n{results}")