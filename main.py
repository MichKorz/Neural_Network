from Network import Network
from Coach import Coach


network = Network("MNIST#6", True, 0.1, 784, [32], 10)
coach = Coach(network, 'Mnist')
coach.train(5, 100)
