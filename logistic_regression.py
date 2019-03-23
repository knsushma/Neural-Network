import numpy as np
import json
import math




class NeuralNetwork:
    def __init__(self, inputFileName, epoch, learning_rate):
        self.inputFile = inputFileName
        self.features = []
        self.dataset = [0][0]
        self.std_dataset = [0][0]
        self.labelTypes = []
        self.labels = []
        self.mean = []
        self.sd = []
        self.weights = []
        self.shape = (0, 0)
        self.epoch = epoch
        self.learning_rate = learning_rate


    def load_and_init_dataset(self):
        dataSet = json.load(open(self.inputFile))
        self.features = np.array(dataSet["metadata"]["features"][0:-1])
        self.dataset = np.array(dataSet["data"])
        self.std_dataset = np.array(dataSet["data"])
        self.labels = self.dataset[:, -1]
        self.labelTypes = self.features[:,1]
        self.shape = self.dataset.shape

    def compute_mean_and_sd(self, data_set):
        self.mean = np.mean(data_set, axis=0)
        self.sd = np.std(data_set, axis=0)
        self.sd[self.sd == 0.00] = 1.0


    def standardize_dataset(self, dataset_obj):
        numeric_indices = []
        categorical_indices = []
        for index in range(len(self.features)):
            if (self.features[index][1] == "numeric"):
                numeric_indices.append(index)
            else:
                categorical_indices.append(index)

        if (len(numeric_indices)):
            self.compute_mean_and_sd(self.std_dataset[:, numeric_indices].astype(float))
            self.std_dataset[:,numeric_indices] = (self.std_dataset[:,numeric_indices].astype(float) - self.mean) / self.sd
            dataset_obj.std_dataset[:,numeric_indices] = (dataset_obj.std_dataset[:,numeric_indices].astype(float) - self.mean) / self.sd

        catrgorical_dataset = np.zeros((dataset_obj.shape[0], len(categorical_indices)))
        if (len(categorical_indices)):
            trainMatrix = self.dataset[:, categorical_indices]
            testMatrix = dataset_obj.dataset[:, categorical_indices]

        return dataset_obj


    def add_biased_unit(self):
        add_one_bias = np.ones((self.shape[0], 1))
        self.std_dataset = np.hstack((add_one_bias, self.std_dataset))

    def flatten(self, irregular_list):
        newlist = []
        for item in irregular_list:
            if isinstance(item, list):
                newlist = newlist + flatten(item)
            else:
                newlist.append(item)
        return newlist


    def train_model(self, epoch):
        # model_net_values = np.sum(self.std_dataset[:,0:-1].astype(float) * self.weights, axis=1)
        # model_activaton_values = 1.0 / ( 1 + np.exp(-model_net_values))
        # cross_entropy_error = np.sum(1.0/2.0 * np.square((self.std_dataset[:,-1].astype(float)-model_activaton_values)))
        # correct_classifications = np.sum(np.where(model_activaton_values>=0.5, 1, 0))
        # miss_classifications = self.shape[0] - correct_classifications
        # print(epoch, " ", correct_classifications, " ", miss_classifications)
        # #print("testing", (self.std_dataset[:,-1].astype(float) - model_activaton_values))
        # gradients = -1.0 * (self.std_dataset[:,-1].astype(float) - model_activaton_values) * (model_activaton_values) * (1 - model_activaton_values) * (self.std_dataset[:,0:-1].astype(float))
        # #self.weights = self.weights + (-1.0 * self.learning_rate * gradients)
        # print()

        cross_entropy_error = 0.0
        corrects = 0
        for row in self.std_dataset.astype(float):
            row_attributes = row[0: -1]
            neural_output = self.find_neural_net_output(self.weights, row_attributes)
            sigmoid_activation_value = 1.0 / (1 + math.exp(-neural_output))
            if ((sigmoid_activation_value >= 0.5 and row[-1]==1.0) or  (sigmoid_activation_value < 0.5 and row[-1]==0.0)):
                corrects += 1
            cross_entropy_error += -1.0 * row[-1] * math.log(sigmoid_activation_value) - (1-row[-1]) * math.log((1-sigmoid_activation_value))
            for index in range(len(row_attributes)):
                gradient =  (sigmoid_activation_value - row[-1]) * row_attributes[index]
                self.weights[index] += -1.0 * self.learning_rate * gradient
        print("{0} {1:.12f} {2} {3}".format(epoch, cross_entropy_error, corrects, (self.shape[0]-corrects)))

    def find_neural_net_output(self, weights, x_attributes):
        sum = 0.0
        for index in range(len(x_attributes)):
            sum += (weights[index]) * (x_attributes[index])
        return sum

if __name__ == '__main__':
    np.random.seed(0)
    train_neural = NeuralNetwork("./Resources/banknote_train.json", 10, 0.01)
    test_neural = NeuralNetwork("./Resources/banknote_test.json", 10, 0.01)

    train_neural.load_and_init_dataset()
    test_neural.load_and_init_dataset()

    train_neural.standardize_dataset(test_neural)
    train_neural.add_biased_unit()

    input_units_size = len(train_neural.flatten(train_neural.labelTypes))
    train_neural.weights = np.random.uniform(low=-0.01, high=0.01, size=(1, input_units_size+1)).tolist()[0]

    for r in range(1, train_neural.epoch+1):
        train_neural.train_model(r)