import numpy as np
import json
import math
import sys

class NeuralNetwork:
    def __init__(self, learning_rate, max_epoch, inputFileName):
        self.input_file = inputFileName
        self.features = []
        self.dataset = [0][0]
        self.std_dataset = [0][0]
        self.label_types = []
        self.labels = []
        self.label_class = []
        self.mean = []
        self.sd = []
        self.weights = []
        self.shape = (0, 0)
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate

    def load_and_init_dataset(self):
        dataset = json.load(open(self.input_file))
        self.features = np.array(dataset["metadata"]["features"][0:-1])
        self.dataset = np.array(dataset["data"])
        self.std_dataset = np.array(dataset["data"], dtype=object)
        self.labels = self.dataset[:, -1]
        self.label_types = self.features[:, 1]
        self.label_class = dataset["metadata"]["features"][-1][1]
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

        if (len(categorical_indices)):
            for index in categorical_indices:
                for row_index, row in enumerate(self.std_dataset):
                    self.std_dataset[row_index, index] = np.isin(self.features[index][1], self.std_dataset[row_index, index]).astype(int).tolist()
                for row_index, row in enumerate(dataset_obj.std_dataset):
                    dataset_obj.std_dataset[row_index, index] = np.isin(self.features[index][1], dataset_obj.std_dataset[row_index, index]).astype(int).tolist()

        if (len(numeric_indices)):
            self.compute_mean_and_sd(self.std_dataset[:, numeric_indices].astype(float))
            self.std_dataset[:,numeric_indices] = (self.std_dataset[:,numeric_indices].astype(float) - self.mean) / self.sd
            dataset_obj.std_dataset[:,numeric_indices] = (dataset_obj.std_dataset[:,numeric_indices].astype(float) - self.mean) / self.sd

    def train_model(self, epoch):
        cross_entropy_error = 0.0
        corrects = 0
        for row_index, row in enumerate(self.std_dataset):
            if (row[-1] == self.label_class[1]):
                y = 1
            else:
                y = 0
            row_attributes = self.flatten(row[0:-1])
            w_i_dot_product = np.dot(self.weights, row_attributes)
            sigmoid_activation_value = 1.0 / (1 + math.exp(-w_i_dot_product))
            if ((sigmoid_activation_value >= 0.5 and y==1) or  (sigmoid_activation_value < 0.5 and y==0)):
                corrects += 1
            cross_entropy_error += (-1.0 * y * math.log(sigmoid_activation_value)) - ((1.0-y) * math.log((1-sigmoid_activation_value)))
            for index in range(len(row_attributes)):
                self.weights[index] += self.learning_rate * (y - sigmoid_activation_value) * row_attributes[index]
        print("{0} {1:.12f} {2} {3}".format(epoch, cross_entropy_error, corrects, (self.shape[0]-corrects)))

    def prediction_on_testdate(self, test_obj):
        corrects = 0
        TP = 0
        FP = 0

        label_list = self.get_binary_label_list(test_obj.std_dataset[:, -1])
        for index,row in enumerate(test_obj.std_dataset):
            row_attributes = self.flatten(row[0:-1])
            w_i_dot_product = np.dot(self.weights, self.flatten(row_attributes))
            sigmoid_activation_value = 1.0 / (1 + math.exp(-w_i_dot_product))
            actual_class = label_list[index]
            predicted_class = int(sigmoid_activation_value >= 0.5)
            if ((sigmoid_activation_value >= 0.5 and label_list[index]==1) or  (sigmoid_activation_value < 0.5 and label_list[index]==0)):
                corrects += 1
            if (sigmoid_activation_value >= 0.5 and label_list[index]==1):
                TP += 1
            if (sigmoid_activation_value >= 0.5 and label_list[index]==0):
                FP += 1
            print("{0:.12f} {1} {2}".format(sigmoid_activation_value, predicted_class, actual_class))
        print("{0} {1}".format(corrects, (test_obj.shape[0]-corrects)))

        F1_score = 0.0
        if (TP + FP != 0):
            precision = TP / (TP + FP)
            recall = TP / (np.sum(label_list))
            F1_score = (2 * precision * recall) / (precision + recall)
        print("{0:.12f}".format(F1_score))

    def get_binary_label_list(self, dataset_labels):
        label_list = []
        for label in dataset_labels:
            if (self.label_class[0] == label):
                label_list.append(0)
            else:
                label_list.append(1)

        return label_list

    def add_biased_unit(self):
        add_one_bias = np.ones((self.shape[0], 1))
        self.std_dataset = np.hstack((add_one_bias, self.std_dataset))

    def flatten(self, irregular_list):
        newlist = []
        for item in irregular_list:
            if isinstance(item, list):
                newlist = newlist + self.flatten(item)
            else:
                newlist.append(item)
        return newlist


if __name__ == '__main__':
    np.random.seed(0)

    if (len(sys.argv)<5):
        print("Please pass 4 arguments. 1) Learning Rate 2) Epochs 3) Training File Path, 4) Testing File path ")
        sys.exit(1)

    learning_rate = float(sys.argv[1])
    max_epoch = int(sys.argv[2])
    train_file = sys.argv[3]
    test_file = sys.argv[4]

    train_neural = NeuralNetwork(learning_rate, max_epoch, train_file)
    test_neural = NeuralNetwork(learning_rate, max_epoch, test_file)

    train_neural.load_and_init_dataset()
    test_neural.load_and_init_dataset()

    train_neural.standardize_dataset(test_neural)
    train_neural.add_biased_unit()
    test_neural.add_biased_unit()

    input_units_size = len(train_neural.flatten(train_neural.label_types))
    train_neural.weights = np.random.uniform(low=-0.01, high=0.01, size=(1, input_units_size+1)).tolist()[0]

    for e in range(1, train_neural.max_epoch + 1):
        train_neural.train_model(e)

    train_neural.prediction_on_testdate(test_neural)