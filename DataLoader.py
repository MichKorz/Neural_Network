class DataLoader:

    def load_data_set(self, name):
        if name == "Iris":
            return self.load_iris()
        elif name == "Mnist":
            return self.load_mnist()
        else:
            print('Data set: '+name+' not found')

    @staticmethod
    def load_iris():
        path = 'Iris/bezdekIris.data'

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        with open(path, 'r') as file:
            for i in range(3):
                for j in range(40):
                    line = file.readline().strip().split(',')
                    inputs = [float(line[x]) for x in range(4)]
                    answer = [1 if x == i else 0 for x in range(3)]
                    train_data.append(inputs)
                    train_labels.append(answer)

                for k in range(10):
                    line = file.readline().strip().split(',')
                    inputs = [float(line[x]) for x in range(4)]
                    answer = [1 if x == i else 0 for x in range(3)]
                    test_data.append(inputs)
                    test_labels.append(answer)

        return train_data, train_labels, test_data, test_labels

    @staticmethod
    def load_mnist():
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        with open('MNIST/train_data.data', 'r') as file:
            for line in file:
                raw = line.strip().split(',')
                inputs = [float(x) for x in raw]
                train_data.append(inputs)

        with open('MNIST/train_labels.data', 'r') as file:
            for line in file:
                raw = line.strip().split(',')
                inputs = [float(x) for x in raw]
                train_labels.append(inputs)

        with open('MNIST/test_data.data', 'r') as file:
            for line in file:
                raw = line.strip().split(',')
                inputs = [float(x) for x in raw]
                test_data.append(inputs)

        with open('MNIST/test_labels.data', 'r') as file:
            for line in file:
                raw = line.strip().split(',')
                inputs = [float(x) for x in raw]
                test_labels.append(inputs)

        return train_data, train_labels, test_data, test_labels
