from Network import Network
from Coach import Coach
from DataLoader import DataLoader
from ParametersHandler import ParametersHandler
import sys


if len(sys.argv) == 1:
    network = Network("MNIST#7", False, 0.5, 784, [12], 10)
    data_loader = DataLoader("MNIST/train_data.data", "MNIST/train_labels.data",
                             "MNIST/test_data.data", "MNIST/test_labels.data")
    coach = Coach(network, data_loader, True)

    coach.train(5, 100)
    sys.exit(0)

ph = ParametersHandler(sys.argv)
network = Network(ph.name, ph.load, ph.learning_rate, ph.input_size, ph.h_layers, ph.output_size)
data_loader = DataLoader("C:/Users/Admin/PycharmProjects/NeuralNetwork/MNIST/train_data.data",
                         "C:/Users/Admin/PycharmProjects/NeuralNetwork/MNIST/train_labels.data",
                         "C:/Users/Admin/PycharmProjects/NeuralNetwork/MNIST/test_data.data",
                         "C:/Users/Admin/PycharmProjects/NeuralNetwork/MNIST/test_labels.data")
coach = Coach(network, data_loader, ph.variable_learning_rate)

print("Setup compleate, begining training!", flush=True)

coach.train(ph.epochs, ph.samples_per_epoch)
sys.exit(0)
