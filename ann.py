import csv, random, math
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import statistics


class Neural_Nets():

    def loadData(self, filename):
        lines = csv.reader(open(filename, "rt"))
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
        return dataset

    # This methods sets the data for the model
    # So it will be changes according to the data structure and types.
    def train_test_data(self, dataset, splitRatio):
        trainData = int(len(dataset) * splitRatio)
        trainSet = []
        copy = list(dataset)
        while len(trainSet) < trainData:
            index = random.randrange(len(copy))
            trainSet.append(copy.pop(index))
        X_train = [trainSet[i][:-1] for i in range(len(trainSet)) ]
        y_train = [trainSet[i][-1] for i in range(len(trainSet)) ]
        X_test = [copy[i][:-1] for i in range(len(copy)) ]
        y_test = [copy[i][-1] for i in range(len(copy)) ]
        return X_train, y_train, X_test, y_test

    def standardize_data(self, dataset):
        transform = zip(*dataset)
        stats = map(lambda x: [sum(x)/len(x), statistics.stdev(x)], transform)
        stats = list(stats)
        standardized_data = []
        for i in range(len(dataset)):
            temp_list = []
            for j in range(len(dataset[i])):
                standardized_val = (dataset[i][j]-stats[j][0])/(stats[j][1])
                temp_list.append(standardized_val)
            standardized_data.append(temp_list)
        return standardized_data

    # Sigmoid Function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of Sigmoid Function
    def derivatives_sigmoid(self, x):
        return x * (1 - x)


    def fit(self, x_train, y_train, lr, wh, bh, wout, bout ):
        for i in range(epoch):
            # Forward propagation
            z1 = np.array(x_train).dot(wh) + bh
            activations = self.sigmoid(z1)
            z2 = activations.dot(wout) + bout
            output = self.sigmoid(z2)

            # Backpropagation
            y_train = np.array(y_train).reshape((614,1))
            error = np.array(y_train) - output
            d_output_layer = self.derivatives_sigmoid(output)
            d_hidden_layer = self.derivatives_sigmoid(activations)
            d_output = error * d_output_layer
            error_hidden_layer = d_output.dot(wout.T)
            delta_hidden_layer = error_hidden_layer * d_hidden_layer

            # Weights and bias update using gradient descent
            wout += activations.T.dot(d_output) * lr
            bout += np.sum(d_output, axis=0, keepdims=True) * lr
            wh += np.array(x_train).T.dot(delta_hidden_layer) * lr
            bh += np.sum(delta_hidden_layer, axis=0, keepdims=True) * lr

        self.model = {'W1': wh, 'b1': bh, 'W2': wout, 'b2': bout}


    def predict(self,  x):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # Forward propagation
        z1 = np.array(x).dot(W1) + b1
        activations = self.sigmoid(z1)
        z2 = activations.dot(W2) + b2
        output = self.sigmoid(z2)
        labels = [1 if x > 0.5 else 0 for x in output  ]
        return labels




if __name__ == '__main__':
    datafile = 'D:/blog/random_datasets/pima-indians-diabetes.csv'
    learning_rate = 0.1
    reg_lambda = 0.01
    hidden_layers = 3
    epoch = 10000 # Number of iterations
    ann = Neural_Nets()
    dataset = ann.loadData(datafile)
    X_train, y_train, X_test, y_test = ann.train_test_data(dataset, 0.8)
    X_train, X_test = ann.standardize_data(X_train), ann.standardize_data(X_test)
    input_dimensions = len(X_train[0])
    np.random.seed(0) # Setting constant to get similar random numbers every time
    init_weights_hidden_layer = np.random.randn(input_dimensions, hidden_layers) / np.sqrt(input_dimensions)
    init_bias_hidden_layer = np.zeros((1, hidden_layers))
    init_weights_output_layer = np.random.randn(hidden_layers, 1) # 1 because output layer contains 1 output
    init_bias_output_layer = np.zeros((1, 1))
    ann.fit(X_train, y_train, learning_rate, init_weights_hidden_layer,
            init_bias_hidden_layer, init_weights_output_layer, init_bias_output_layer )
    y_predicted = ann.predict(X_test)
    accuray = accuracy_score(y_test, y_predicted)*100
    print(ann.model)
    print('Accuracy of the model is {0}%'.format(math.ceil(accuray*100)/100))


