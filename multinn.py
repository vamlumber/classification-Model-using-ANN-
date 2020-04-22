# Patil, Abhishek
# 1001-668-197
# 2019-10-28
# Assignment-03-01

# using tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each the input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None
        self.number_of_neurons_list = []

    def add_layer(self, num_nodes, activation_function):
        self.number_of_neurons_list.append(num_nodes)
        self.activations.append(activation_function)
        if len(self.weights)==0:
            self.weights.append(np.random.randn(self.input_dimension,num_nodes))
        else:
            self.weights.append(np.random.randn(self.weights[-1].shape[1],num_nodes))
        self.biases.append(np.random.randn(1,num_nodes))

    def get_weights_without_biases(self, layer_number):
        return self.weights[layer_number]        

    def get_biases(self, layer_number):
        return self.biases[layer_number]
        
    def set_weights_without_biases(self, weights, layer_number):
        self.weights[layer_number]= weights
        
    def set_biases(self, biases, layer_number):
        self.biases[layer_number] = biases
        
    def set_loss_function(self, loss_fn):
        self.loss = loss_fn

    def sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def linear(self, x):
        return x

    def relu(self, x):
        out = tf.nn.relu(x)
        return out

    def cross_entropy_loss(self, y, y_hat):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def predict(self, X):
        X_test = X
        for layer_number in range(len(self.number_of_neurons_list)):#Forward Feed
            X_test = self.activations[layer_number](tf.add(tf.matmul(X_test,self.weights[layer_number]),self.biases[layer_number]))
        return X_test

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8, regularization_coeff=1e-6):
        # slice_data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        # slice_data = slice_data.batch(batch_size)
        for epoch in range(0,num_epochs):
            q =int(X_train.shape[0]/batch_size)
            if(X_train.shape[0]%batch_size != 0):
                q+= 1
            for i in range(q):
                with tf.GradientTape() as tape:
                    X_batch = X_train[batch_size*i:batch_size*(i+1)]
                    y_hat = None
                    for layer_number in range(len(self.number_of_neurons_list)):#Forward Feed
                        if layer_number == 0:
                            net_value = tf.add(tf.matmul(X_batch,self.weights[layer_number]),self.biases[layer_number])
                        else:
                            net_value = tf.add(tf.matmul(y_hat,self.weights[layer_number]),self.biases[layer_number])
                        y_hat = self.activations[layer_number](net_value)
                    loss = self.loss(y_train[batch_size*i:batch_size*(i+1)],y_hat)
                    dw,db = tape.gradient(loss,[self.weights,self.biases])
                    for gk in range(len(self.number_of_neurons_list)):
                        self.weights[gk].assign_sub(alpha * dw[gk])
                        self.biases[gk].assign_sub(alpha * db[gk])
                

    def calculate_percent_error(self, X, y):
        y_hat = self.predict(X)
        a = np.argmax(y_hat,axis=1)
        false_predict = 0
        for y_one,act_one in zip(y,a):
            if(y_one != act_one):
                false_predict += 1
        percent_error = false_predict/y.shape[0]
        return percent_error

    def calculate_confusion_matrix(self, X, y):
        y_hat = self.predict(X)
        a = np.argmax(y_hat,axis=1)
        cofusion_matrix = np.zeros((y_hat.shape[1],y_hat.shape[1]))
        for y_one,act_one in zip(y,a):
            cofusion_matrix[y_one][act_one] += 1
        return cofusion_matrix

if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist

    np.random.seed(seed=1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape and Normalize data
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]
    indices = list(range(X_train.shape[0]))
    # np.random.shuffle(indices)
    number_of_samples_to_use = 500
    X_train = X_train[indices[:number_of_samples_to_use]]
    y_train = y_train[indices[:number_of_samples_to_use]]
    multi_nn = MultiNN(input_dimension)
    number_of_classes = 10
    activations_list = [multi_nn.sigmoid, multi_nn.sigmoid, multi_nn.linear]
    number_of_neurons_list = [50, 20, number_of_classes]
    for layer_number in range(len(activations_list)):
        multi_nn.add_layer(number_of_neurons_list[layer_number], activation_function=activations_list[layer_number])
    for layer_number in range(len(multi_nn.weights)):
        W = multi_nn.get_weights_without_biases(layer_number)
        W = tf.Variable((np.random.randn(*W.shape)) * 0.1, trainable=True)
        multi_nn.set_weights_without_biases(W, layer_number)
        b = multi_nn.get_biases(layer_number=layer_number)
        b = tf.Variable(np.zeros(b.shape) * 0, trainable=True)
        multi_nn.set_biases(b, layer_number)
    multi_nn.set_loss_function(multi_nn.cross_entropy_loss)
    percent_error = []
    for k in range(10):
        multi_nn.train(X_train, y_train, batch_size=100, num_epochs=20, alpha=0.8)
        percent_error.append(multi_nn.calculate_percent_error(X_train, y_train))
    confusion_matrix = multi_nn.calculate_confusion_matrix(X_train, y_train)
    print("Percent error: ", np.array2string(np.array(percent_error), separator=","))
    print("************* Confusion Matrix ***************\n", np.array2string(confusion_matrix, separator=","))
