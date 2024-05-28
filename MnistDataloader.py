import numpy as np  # linear algebra
import struct
from array import array

class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

        train_data = []
        for i in x_train:
            image = []
            for j in i:
                for k in j:
                    image.append(float(k) / 256)
            train_data.append(image)

        train_labels = []
        for i in range(len(y_train)):
            train_labels.append([1 if j == y_train[i] else 0 for j in range(0, 10)])

        test_data = []
        for i in x_test:
            image = []
            for j in i:
                for k in j:
                    image.append(float(k) / 256)
            test_data.append(image)

        test_labels = []
        for i in range(len(y_test)):
            test_labels.append([1 if j == y_test[i] else 0 for j in range(0, 10)])

        with open("MNIST/train_data.data", 'w') as file:
            for i in range(len(train_data)):
                for j in range(len(train_data[i])):
                    file.write(str(train_data[i][j]))
                    if j < len(train_data[i])-1:
                        file.write(',')
                file.write('\n')

        with open("MNIST/train_labels.data", 'w') as file:
            for i in range(len(train_labels)):
                for j in range(len(train_labels[i])):
                    file.write(str(train_labels[i][j]))
                    if j < len(train_labels[i])-1:
                        file.write(',')
                file.write('\n')

        with open("MNIST/test_data.data", 'w') as file:
            for i in range(len(test_data)):
                for j in range(len(test_data[i])):
                    file.write(str(test_data[i][j]))
                    if j < len(test_data[i])-1:
                        file.write(',')
                file.write('\n')

        with open("MNIST/test_labels.data", 'w') as file:
            for i in range(len(test_labels)):
                for j in range(len(test_labels[i])):
                    file.write(str(test_labels[i][j]))
                    if j < len(test_labels[i])-1:
                        file.write(',')
                file.write('\n')

        return (train_data, train_labels), (test_data, test_labels)
