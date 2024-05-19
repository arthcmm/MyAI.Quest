import numpy as np


class Perceptron:
    def __init__(self, num_inputs, threshold=100, learning_rate=0.01):
        self.weights = np.zeros(num_inputs + 1)  # sums bias
        self.threshold = threshold
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


def generate_data(func, num_inputs):  # generates the combinations
    inputs = np.array(np.meshgrid(*[[0, 1]] * num_inputs)).T.reshape(-1, num_inputs)
    labels = np.array([func(*input_row) for input_row in inputs])
    return inputs, labels


and_func = lambda *inputs: int(all(inputs))
or_func = lambda *inputs: int(any(inputs))

# asks the user for an input
num_inputs = int(input("Digite a quantidade de entradas (2, 3, 4... n): "))
func_type = input("Digite a função desejada (AND ou OR): ").strip().upper()

if func_type == 'AND':
    func = and_func
elif func_type == 'OR':
    func = or_func
else:
    raise ValueError("Função não suportada. Escolha AND ou OR.")

# gets train data
inputs, labels = generate_data(func, num_inputs)

# creates the perceptron and call the training
perceptron = Perceptron(num_inputs=num_inputs)
perceptron.train(inputs, labels)

# test the combinations
for input_row in inputs:
    output = perceptron.predict(input_row)
    print(f"Input: {input_row}, Output {func_type}: {output}")
