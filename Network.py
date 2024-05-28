import random


class Network:

    def __init__(self, name, load, learning_rate=None, inputs=None, h_layers=None, outputs=None):
        self.name = name
        self.learning_rate = 0.1
        self.weights = []
        self.biases = []

        self.total_weight_gradient = []
        self.total_bias_gradient = []

        if load:
            self.load_from_file()
            return

        self.learning_rate = learning_rate

        self.initialize_backprop(inputs, h_layers, outputs)

        layer = [[random.uniform(-1, 1) for _ in range(inputs)] for _ in range(h_layers[0])]
        self.weights.append(layer)  # weights from input layer

        for i in range(0, len(h_layers)-1):
            layer = [[random.uniform(-1, 1) for _ in range(h_layers[i])] for _ in range(h_layers[i+1])]
            self.weights.append(layer)  # weights between hidden layers

        layer = [[random.uniform(-1, 1) for _ in range(h_layers[-1])] for _ in range(outputs)]
        self.weights.append(layer)  # weights before output layer

        for i in range(0, len(h_layers)):
            b_set = [random.uniform(-1, 1) for _ in range(0, h_layers[i])]

            self.biases.append(b_set)
        self.biases.append([random.uniform(-1, 1) for _ in range(0, outputs)])

    def initialize_backprop(self, inputs, h_layers, outputs):

        self.total_weight_gradient.append([[0.0 for _ in range(inputs)] for _ in range(h_layers[0])])
        for i in range(0, len(h_layers) - 1):
            self.total_weight_gradient.append([[0.0 for _ in range(h_layers[i])] for _ in range(h_layers[i + 1])])
        self.total_weight_gradient.append([[0.0 for _ in range(h_layers[-1])] for _ in range(outputs)])

        for i in range(0, len(h_layers)):
            self.total_bias_gradient.append([0.0 for _ in range(0, h_layers[i])])
        self.total_bias_gradient.append([0.0 for _ in range(0, outputs)])

    def save_to_file(self):
        with open(self.name + '.txt', 'w') as file:
            file.write(str(len(self.weights[0][0])) + '\n')  # inputs
            file.write(str(len(self.weights[-1])) + '\n')  # outputs
            for i in range(0, len(self.weights)-1):
                file.write(str(len(self.weights[i])) + ' ')  # length of hidden layers
            file.write('\n')

            file.write('123\n')  # set magic number

            for i in self.weights:
                for j in i:
                    for k in j:
                        file.write(str(k) + ' ')
                    file.write('\n')

            file.write('456\n')  # set magic number

            for i in self.biases:
                for j in i:
                    file.write(str(j) + ' ')
                file.write('\n')

            file.write('789\n')  # set magic number

            file.write(str(self.learning_rate))

    def load_from_file(self):
        with open(self.name + '.txt', 'r') as file:
            inputs = int(file.readline())
            outputs = int(file.readline())
            h_layers = [int(x) for x in file.readline().strip().split(' ')]

            magic_number = file.readline().strip()
            if magic_number != '123':
                print('magic number mismatch after: Initial Values\n')
                print('number read: ' + str(magic_number) + '\n')
                return

            self.initialize_backprop(inputs, h_layers, outputs)

            layer = []
            for i in range(0, h_layers[0]):
                layer.append([float(x) for x in file.readline().strip().split(' ')])  # input weight layer
            self.weights.append(layer)

            for i in range(0, len(h_layers)-1):
                layer = []
                for j in range(0, h_layers[i+1]):
                    layer.append([float(x) for x in file.readline().strip().split(' ')])  # hidden weight layers
                self.weights.append(layer)

            layer = []
            for i in range(0, outputs):
                layer.append([float(x) for x in file.readline().strip().split(' ')])  # output weight layer
            self.weights.append(layer)

            magic_number = file.readline().strip()
            if magic_number != '456':
                print('magic number mismatch after: Initial Values\n')
                print('number read: ' + str(magic_number) + '\n')
                return

            for i in range(0, len(h_layers)+1):
                self.biases.append([float(x) for x in file.readline().strip().split(' ')])  # biases

            magic_number = file.readline().strip()
            if magic_number != '789':
                print('magic number mismatch after: Initial Values\n')
                print('number read: ' + str(magic_number) + '\n')
                return

            self.learning_rate = float(file.readline().strip())

    def forward_pass(self, data, save_hidden):
        inputs = data

        h_layers = []

        for i in range(0, len(self.weights)):
            output = []
            for j in range(0, len(self.weights[i])):
                value = 0.0
                for k in range(0, len(inputs)):
                    value += (inputs[k] * self.weights[i][j][k])
                value += self.biases[i][j]
                value = self.sigmoid(value)
                output.append(value)
            h_layers.append(output)
            inputs = output

        h_layers.pop()

        if save_hidden:
            return inputs, h_layers
        else:
            return inputs

    def backpropagation(self, inp, ans):  # inp: [], ans: []

        weight_gradient = []
        bias_gradient = []

        outputs, x = self.forward_pass(inp, True)
        h_layers = [inp]
        for o in x:
            h_layers.append(o)

        costs = []
        for i in range(0, len(ans)):
            costs.append(0.5*((ans[i]-outputs[i])**2))

        # Output Layer

        effects = []
        for i in range(1, len(h_layers) + 1):
            effects.insert(0, [0.0 for _ in range(len(h_layers[-i]))])

        weights_matrix = []
        bias_layer = []
        for j in range(0, len(ans)):
            delta = -1 * (ans[j] - outputs[j]) * outputs[j] * (1 - outputs[j])
            bias_layer.append(delta)
            weights_layer = []
            for k in range(0, len(h_layers[-1])):
                weights_layer.append(h_layers[-1][k]*delta)
                effects[-1][k] += self.weights[-1][j][k]*delta
            weights_matrix.append(weights_layer)

        weight_gradient.append(weights_matrix)
        bias_gradient.append(bias_layer)

        # i-layer

        weights_matrix = []
        bias_layer = []
        for i in range(1, len(self.weights)):
            for j in range(0, len(h_layers[-i])):
                delta = effects[-i][j]*h_layers[-i][j]*(1-h_layers[-i][j])
                bias_layer.append(delta)
                layer = []
                for k in range(0, len(h_layers[-i-1])):
                    layer.append(delta*h_layers[-i-1][k])
                    effects[-i-1][k] += self.weights[-i-1][j][k]*delta
                weights_matrix.append(layer)

            weight_gradient.insert(0, weights_matrix)
            weights_matrix = []
            bias_gradient.insert(0, bias_layer)
            bias_layer = []

        # TODO Naprawić ten syf

        for i in range(0, len(self.weights)):
            for j in range(0, len(self.weights[i])):
                for k in range(0, len(self.weights[i][j])):
                    self.total_weight_gradient[i][j][k] += weight_gradient[i][j][k]

        for i in range(0, len(self.biases)):
            for j in range(0, len(self.biases[i])):
                self.total_bias_gradient[i][j] += bias_gradient[i][j]

    def sigmoid(self, value):
        e = 2.71828182
        x = (e**value)/(1+e**value)
        return x

    def apply_gradient(self, direction):

        for i in range(0, len(self.weights)):
            for j in range(0, len(self.weights[i])):
                for k in range(0, len(self.weights[i][j])):
                    self.weights[i][j][k] -= self.total_weight_gradient[i][j][k]*self.learning_rate*direction

        for i in range(0, len(self.biases)):
            for j in range(0, len(self.biases[i])):
                self.biases[i][j] -= self.total_bias_gradient[i][j]*self.learning_rate*direction

    def clear_gradient(self):
        for i in range(0, len(self.weights)):
            for j in range(0, len(self.weights[i])):
                for k in range(0, len(self.weights[i][j])):
                    self.total_weight_gradient[i][j][k] = 0

        for i in range(0, len(self.biases)):
            for j in range(0, len(self.biases[i])):
                self.total_bias_gradient[i][j] = 0
