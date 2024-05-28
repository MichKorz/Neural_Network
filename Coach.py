import random
from DataLoader import DataLoader


class Coach:

    def __init__(self, network, data_loader, varaible_learning_rate):
        self.network = network
        self.data_loader = data_loader
        self.train_data, self.train_labels, self.test_data, self.test_labels = self.data_loader.load_data()
        self.varaible_learning_rate = varaible_learning_rate

    def calculate_cost(self):
        total_cost = 0
        for i in range(0, len(self.test_data)):
            cost = 0
            output = self.network.forward_pass(self.test_data[i], False)
            for j in range(0, len(self.test_labels[i])):
                cost += 0.5*((self.test_labels[i][j] - output[j])**2)
            total_cost += cost
        total_cost /= len(self.test_data)
        return total_cost

    def calculate_accuracy(self):
        guessed_right = 0
        for i in range(0, len(self.test_data)):
            output = self.network.forward_pass(self.test_data[i], False)
            pick = -1
            certainty = 0
            for j in range(len(output)):
                if output[j] > certainty:
                    certainty = output[j]
                    pick = j
            ans = 0
            for j in range(len(self.test_labels[i])):
                if self.test_labels[i][j] == 1:
                    ans = j
                    break
            if pick == ans:
                guessed_right += 1
        percentage = guessed_right/len(self.test_labels) * 100
        return percentage

    def train(self, passes, samples_per_pass):

        cost = self.calculate_cost()
        print(str(cost) + ' : ' + str(self.calculate_accuracy()) + '%', flush=True)

        samples_left = [x for x in range(len(self.train_data))]

        for i in range(passes):
            for j in range(samples_per_pass):
                choices = len(samples_left)
                if choices == 0:
                    samples_left = [x for x in range(len(self.train_data))]
                    choices = len(samples_left)
                index = random.randint(0, choices-1)
                sample_id = samples_left[index]
                samples_left.pop(index)
                self.network.backpropagation(self.train_data[sample_id], self.train_labels[sample_id])

            if self.varaible_learning_rate:
                self.network.apply_gradient(1)
                new_cost = self.calculate_cost()
                if new_cost < cost:
                    self.network.learning_rate = self.network.learning_rate * 1.05
                else:
                    print("Decreasing learning rate", flush=True)
                    while new_cost > cost:
                        self.network.apply_gradient(-1)
                        self.network.learning_rate = self.network.learning_rate * 0.8
                        self.network.apply_gradient(1)
                        new_cost = self.calculate_cost()
                self.network.clear_gradient()
                cost = new_cost
                print(str(cost) + ' : ' + str(self.calculate_accuracy()) + '%' + ', epoch: ' + str(i+1), flush=True)
            else:
                self.network.apply_gradient(1)
                self.network.clear_gradient()
                cost = self.calculate_cost()
                print(str(cost) + ' : ' + str(self.calculate_accuracy()) + '%' + ', epoch: ' + str(i+1), flush=True)

        self.network.save_to_file()


