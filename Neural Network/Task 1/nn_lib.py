import numpy as np
import pickle
import math
import random
import sys


def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative log-
    likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, x):
        self._cache_current = x
        # Apply sigmoid function, vectorized
        sigma = np.vectorize(lambda x : 1 / (1 + math.exp(-x)))
        return sigma(x)

    def backward(self, grad_z):
        sigma = lambda x : (1 / (1 + math.exp(-x)))
        deriv_sigma = np.vectorize(lambda x : sigma(x) - sigma(x)**2)

        return np.multiply(grad_z, deriv_sigma(self._cache_current))

    def update_params(self, learning_rate):
        pass


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        self._cache_current = None


    def forward(self, x):
        self._cache_current = x
        relu = np.vectorize(lambda x : max(0, x))
        return relu(x)

    def backward(self, grad_z):
        relu_deriv = np.vectorize(lambda x : 1 if x > 0 else 0)
        return np.multiply(grad_z, relu_deriv(self._cache_current))

    def update_params(self, learning_rate):
        pass



class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """Constructor.
        Arguments:
            n_in {int} -- Number (or dimension) of inputs.
            n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        self._W = xavier_init((n_in, n_out), 1)
        self._b = xavier_init((1, self.n_out), 1)

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None


    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).
        Logs information needed to compute gradient at a later stage in
        `_cache_current`.
        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).
        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        self._cache_current = x
        return np.add(np.dot(x, self._W), np.repeat(self._b, len(x), axis=0))


    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).
        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).
        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        # Compute and save dl/dw and dl/db
        self._grad_W_current = np.dot(np.transpose(self._cache_current), grad_z)
        self._grad_b_current = np.dot(np.ones(len(grad_z)), grad_z)

        # Compute dl/dx and propagate backwards
        dl_dx = np.dot(grad_z, np.transpose(self._W))
        return dl_dx


    def update_params(self, learning_rate=0.01):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.
        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """

        self._W -= self._grad_W_current * learning_rate
        self._b -= self._grad_b_current * learning_rate



class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """Constructor.
        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension).
            neurons {list} -- Number of neurons in each layer represented as a
                list (the length of the list determines the number of layers).
            activations {list} -- List of the activation function to use for
                each layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        self._layers = []
        inputSize = self.input_dim

        # Form the multi layer network 
        for i in range(len(activations)):
            outputSize = neurons[i]
            self._layers.append(LinearLayer(inputSize, outputSize))

            if activations[i] == "relu":
                self._layers.append(ReluLayer())
            elif activations[i] == "sigmoid":
                self._layers.append(SigmoidLayer())

            inputSize = outputSize


    def forward(self, x):
        """
        Performs forward pass through the network.
        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).
        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        result = x
        
        # Forward propagate each layer, get result and repeat
        for i in range(len(self._layers)):
            result = self._layers[i].forward(result)

        return result


    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.
        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1,
                #_neurons_in_final_layer).
        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """
        dL_dz = grad_z

        # Given dl/dz, back propagate through each layer
        for layer in reversed(self._layers):
            dL_dz = layer.backward(dL_dz)

        return dL_dz


    def update_params(self, learning_rate=0.01):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.
        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """

        for i in range(0, len(self._layers)):
            self._layers[i].update_params(learning_rate)



def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """Constructor.
        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- Training batch size.
            nb_epoch {int} -- Number of training epochs.
            learning_rate {float} -- SGD learning rate to be used in training.
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag
        self._loss_layer = None

        # Initalize instructed loss function
        if loss_fun == "mse":
            self._loss_layer = MSELossLayer()
        elif loss_fun == "cross_entropy":
            self._loss_layer = CrossEntropyLossLayer()


    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.
        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, ).
        """

        results = self.network.forward(input_dataset)
        ans = self._loss_layer.forward(results, target_dataset)

        return ans


    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.
        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).
        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """
        assert len(input_dataset) == len(target_dataset)
        p = np.random.permutation(len(input_dataset))
        return input_dataset[p], target_dataset[p]


    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.
        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, ).
        """
        rem = len(input_dataset) % self.batch_size

        # If batch size doesn't divide the dataset evenly
        if rem != 0:
            extra = self.batch_size - rem
            # Generate a random offset value
            offset = np.random.randint(extra, high=len(input_dataset))
            # Append subsets of length remainder to the end of the input sets
            input_dataset = np.concatenate((input_dataset, input_dataset[offset-extra : offset]), axis=0)
            target_dataset = np.concatenate((target_dataset, target_dataset[offset-extra : offset]), axis=0)

        for _ in range(self.nb_epoch):
            if self.shuffle_flag:
                input_dataset, target_dataset = self.shuffle(input_dataset, target_dataset)
            
            # Generate training and target set of batch_size 
            batches = np.split(input_dataset, self.batch_size)
            targets = np.split(target_dataset, self.batch_size) 

            # Train and improve on each batch
            for i in range(len(batches)):
                prediction = self.network.forward(batches[i])
                self._loss_layer.forward(prediction, targets[i])
                dl_dz = self._loss_layer.backward()
                self.network.backward(dl_dz)
                self.network.update_params(self.learning_rate)




class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data, high=1, low=0):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)
        Arguments:
            - data {np.ndarray} dataset used to determined the parameters for
            the normalization.
        """
        self.cols = len(data[0])
        self.max = data.max(axis=0)
        self.min = data.min(axis=0)
        self.largest = high
        self.smallest = low


    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.
        Arguments:
            - data {np.ndarray} dataset to be normalized.
        Returns:
            {np.ndarray} normalized dataset.
        """
        for d in data:
            for i in range(len(d)):
                d[i] = self.smallest + ((d[i] - self.min[i]) * (self.largest - self.smallest)) / (self.max[i] - self.min[i])

        return data


    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.
        Arguments:
            - data {np.ndarray} dataset for which to revert normalization.
        Returns:
            {np.ndarray} reverted dataset.
        """
        for d in data:
            for i in range(len(d)):
                d[i] = (((d[i] - self.smallest) * (self.max[i] - self.min[i])) / (self.largest - self.smallest)) + self.min[i]

        return data



def example_main():
    input_dim = 4
    neurons = [128, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4] # Training data
    y = dat[:, 4:] # Results


    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]

    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train, high=1, low=0)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
