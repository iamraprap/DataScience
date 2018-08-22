import numpy as np
class NNet():
    """Implements a basic feedforward neural network."""
    
    def __init__(self):
        self._layers = []  # An ordered list of layers. The first layer is the input; the final is the output.
    
    def _add_layer(self, layer):
        if self._layers:
            # Update pointers. We keep a doubly-linked-list of layers for convenience.
            prev_layer = self._layers[-1]
            prev_layer.set_next_layer(layer)
            layer.set_prev_layer(prev_layer)
            
        self._layers.append(layer)
    
    def add_input_layer(self, size, **kwargs):
        assert type(size).__name__ == 'int', ('Input layer requires integer size. Type was %s instead.' 
                                              % type(size).__name__)
        layer = InputLayer(size=size, **kwargs)
        self._add_layer(layer)

    def add_dense_layer(self, size, **kwargs):
        assert type(size).__name__ == 'int', ('Dense layer requires integer size. Type was %s instead.' 
                                              % type(size).__name__)
        # Find the previous layer's size.
        prev_size = self._layers[-1].size()
        layer = DenseLayer(shape=(prev_size, size), **kwargs)
        self._add_layer(layer)

    def summary(self, verbose=False):
        """Prints a description of the model."""
        for i, layer in enumerate(self._layers):
            print('%d: %s' % (i, str(layer)))
            if verbose:
                print('weights:', layer.get_weights())
                if layer._use_bias:
                    print('bias:', layer._bias)
                print()

    def predict(self, x):
        """Given an input vector x, run it through the neural network and return the output vector."""
        assert isinstance(x, np.ndarray)
        
        output = x
        for layer in self._layers:
            output = layer.feed_forward(output) 
        return output

        
    def train_single_example(self, X_data, y_data, learning_rate=0.01):
        """Train on a single example. X_data and y_data must be numpy arrays."""
        
        assert isinstance(X_data, np.ndarray)
        assert isinstance(y_data, np.ndarray)

        # Forward propagation.
        output = X_data[0]
        for layer in self._layers:
            output = layer.feed_forward(output) 
        
        # Backpropagation.
        error = None
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer._prev_layer._name == "Input":
                break
            if error==None:
                error = self.compute_mean_squared_error(y_data[0], layer._outputs)
            error = layer.backpropagate(error, 0.5)       

    def train(self, X_data, y_data, learning_rate, num_epochs, randomize=True, verbose=True, print_every_n=100):
        """Both X_data and y_data should be ndarrays. One example per row.
        
        This function takes the data and learning rate, and trains the network for num_epochs passes over the 
        complete data set. 
        
        If randomize==True, the X_data and y_data should be randomized at the start of each epoch. Of course,
        matching X,y pairs should have matching indices after randomization, to avoid scrambling the dataset.
        (E.g., a set of indices should be randomized once and then applied to both X and y data.)
        
        If verbose==True, will print a status report every print_every_n epochs with these
        results:
        
        * Results of running "predict" on each example in the training set
        * MSE (mean squared error) on the dataset
        * Accuracy on the dataset
        """
        assert isinstance(X_data, np.ndarray)
        assert isinstance(y_data, np.ndarray)
        assert X_data.shape[0] == y_data.shape[0]

        for epoch in range(num_epochs):
            for row in X_data:
                # Forward propagation.
                output = row
                for layer in self._layers:
                    output = layer.feed_forward(output) 

                # Backpropagation.
                error = None
                for i in reversed(range(len(self._layers))):
                    layer = self._layers[i]
                    if layer._name == "Input":
                        break
                    if error==None:
                        error = self.compute_mean_squared_error(layer._outputs, y_data)
                    error = layer.backpropagate(error, 0.5)

            if epoch%print_every_n==0:
                acc = self.compute_accuracy(X_data, y_data)
                print('>epoch=%d, output=%.3f, mse=%.3f, acc=%.3f' % (epoch, output, error, acc))

    def compute_mean_squared_error(self, X_data, y_data):
        """Given input X_data and target y_data, compute and return the mean squared error."""
        #assert isinstance(X_data, np.ndarray)
        #assert isinstance(y_data, np.ndarray)
        #assert X_data.shape[0] == y_data.shape[0]
        
        return np.square(np.subtract(X_data, y_data)).mean()
    
    def compute_accuracy(self, X_data, y_data):
        """Given input X_data and target y_data, convert outputs to binary using a threshold of 0.5
        and return the accuracy: # examples correct / total # examples."""
        assert isinstance(X_data, np.ndarray)
        assert isinstance(y_data, np.ndarray)
        assert X_data.shape[0] == y_data.shape[0]
        
        correct = 0
        for i in range(len(X_data)):
            outputs = self.predict(X_data[i])
            outputs = outputs > 0.5
            if outputs == y_data[i]:
                correct += 1
        acc = float(correct) / len(X_data)
        return acc

class Activation():  # Do not edit; update derived classes.
    """Base class that represents an activation function and knows how to take its own derivative."""
    def __init__(self, name):
        self.name = name
    
    def activate(x):
        """x is a scalar or a numpy array. Returns the output y, the result of applying the function to input x."""
        raise NotImplementedError()
    
    def derivative_given_y(self, y):
        """y is a scalar or a numpy array. 
        
        Returns the derivative d(f)/dx given the *activation* value y."""
        raise NotImplementedError()

class IdentityActivation(Activation):
    """Activation function that passes input through unchanged."""
    
    def __init__(self):
        super().__init__(name='Identity')
    
    def activate(self, x):
        """x is a scalar or a numpy array. Returns the output y, the result of applying the function to input x."""
        return x
    
    def derivative_given_y(self, y):
        """y is a scalar or a numpy array. 
        
        Returns the derivative d(f)/dx given the *activation* value y."""
        return 1
    
    
class SigmoidActivation(Activation):
    """Sigmoid activation function."""

    def __init__(self):
        super().__init__(name='Sigmoid')
    
    def activate(self, x):
        """x is a scalar or a numpy array. Returns the output y, the result of applying the function to input x."""

        return 1.0 / (1 + np.exp(-x))

    
    def derivative_given_y(self, y):
        """y is a scalar or a numpy array. 
        
        Returns the derivative d(f)/dx given the *activation* value y."""

        return np.dot(y, np.subtract(1.0, y))

def WeightInitializer():
    """Function to return a random weight. for example, return a random float from -1 to 1."""
    return np.random.uniform(-1, 1)


class Layer():
    """Base class for NNet layers. DO NOT MODIFY THIS CLASS. Update derived classes instead.
    
    Conceptually, in this library a Layer consists at a high level of:
      * a collection of weights (a 2D numpy array)
      * the output nodes that come after the weights above
      * the activation function that is applied to the summed signals in these output nodes
      
    So a Layer isn't just nodes -- it's weights as well as nodes.
      
    Specifically, to send signal forward through a 3-layer network, we start with an Input Layer that does
    very little.  The outputs from the Input layer are simply the fed-in input data.  
    
    Then, the next layer will be a Dense layer that holds the weights from the Input layer to the first hidden
    layer and stores the activation function to be used after doing a product of weights and Input-Layer
    outputs.
    
    Finally, another Dense layer will hold the weights from the hidden to the output layer nodes, and stores
    the activation function to be applied to the final output nodes.
    
    For a typical 1-hidden layer network, then, we would have 1 Input layer and 2 Dense layers.
    
    Each Layer also has funcitons to perform the forward-pass and backpropagation steps for the weights/nodes
    associated with the layer.
    
    Finally, each Layer stores pointers to the pervious and next layers, for convenience when implementing
    backprop.
    """
   
    def __init__(self, shape, use_bias, activation_function=IdentityActivation, weight_initializer=None, name=''):
        # These are the weights from the *previous* layer to the current layer.
        self._weights = None
        
        # Tuple of (# inputs, # outputs) for Dense layers or just a scalar for an input layer.
        assert type(shape).__name__ == 'int' or type(shape).__name__ == 'tuple', (
            'shape must be scalar or a 2-element tuple')
        if type(shape).__name__ == 'tuple':
            assert len(shape)==2, 'shape must be 2-dimensional. Was %d instead' % len(shape)
        self._shape = shape 
    
        # True to use a bias node that inputs to each node in this layer; False otherwise.
        self._use_bias = use_bias
        
        if use_bias:
            bias_size = shape[-1] if len(shape) > 1 else shape
            self._bias = np.zeros(bias_size)
            if weight_initializer:
                for i in range(bias_size):
                    self._bias[i] = weight_initializer()
        
        # Activation function to be applied to each dot product of weights with inputs.
        # Instantiate an object of this class.
        self._activation_function = activation_function() if activation_function else None
        
        # Method used to initialize the weights in this Layer at creation time.
        self._weight_initializer = weight_initializer
        
        # Layer name (optional)
        self._name = name
        
        # Calculated output vector from the most recent feed_forward(inputs) call.
        self._outputs = None
        
        # Doubly linked list pointers to neighbor layers.
        self._prev_layer = None  # Previous layer is closer to (or is) the input layer.
        self._next_layer = None  # Next layer is closer to (or is) the output layer.
    
    def set_prev_layer(self, layer):
        """Set pointer to the previous layer."""
        self._prev_layer = layer
    
    def set_next_layer(self, layer):
        """Set pointer to the next layer."""
        self._next_layer = layer
    
    def size(self):
        """Number of nodes in this layer."""
        if type(self._shape).__name__ == 'tuple':
            return self._shape[-1]
        else:
            return self._shape
        
    def get_weights(self):
        """Return a numpy array of the weights for inputs to this layer."""
        return self._weights
    
    def get_bias(self):
        """Return a numpy array of the bias for nodes in this layer."""
        return self._bias
    
    def feed_forward(self, inputs):
        """Feed the given inputs through the input weights and activation function, and set the outputs vector.
        
        Also returns the outputs vector for convenience."""
        raise NotImplementedError()
        
    def backpropagate(self, error, learning_rate):
        """Adjusts the weights coming into this layer based on the given output error vector.
        
        For the output layer, the "error" vector should be a list of output errors, y_k - t_k.
        For a hidden layer, the "error" vector should be a list of the delta values from the following layer, such as delta_z_k
        
        Returns a list of the delta values for each node in this layer. These deltas can be used as the error
        values when calling backpropagate on the previous layer."""
        raise NotimplementedError()
        
    def __str__(self):
        activation_fxn_name = self._activation_function.name if self._activation_function else None
        return '[%s] shape %s, use_bias=%s, activation=%s' % (self._name, self._shape, self._use_bias,
                                                              activation_fxn_name)

class InputLayer(Layer):
    """A neural network 1-dimensional input layer."""
    
    def __init__(self, size, name='Input'):
        assert type(size).__name__ == 'int', 'Input size must be integer. Was %s instead' % type(size).__name__
        super().__init__(shape=size, use_bias=False, name=name, activation_function=None)
        
    def feed_forward(self, inputs):
        #assert len(inputs)==self._shape, 'Inputs must be of size %d; was %d instead' % (self._shape, len(inputs))
        self._outputs = inputs
        return self._outputs

    def backpropagate(self, error, learning_rate):
        return None  # Nothing to do.

class DenseLayer(Layer):
    """A neural network layer that is fully connected to the previous layer."""
    
    def __init__(self, shape, use_bias=True, name='Dense', **kwargs):
        super().__init__(shape=shape, use_bias=use_bias, name=name, **kwargs)
        
        self._weights = np.zeros(shape)
        if self._weight_initializer:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    self._weights[i,j] = self._weight_initializer()
    
    def feed_forward(self, inputs):
        """Feed the given inputs through the input weights and activation function, and set the outputs vector.
        
        Also returns the outputs vector for convenience."""
        #assert len(inputs)==self._shape, 'Inputs must be of size %d; was %d instead' % (self._shape, len(inputs))       
        # Update output vector for later use, and return it.
        self._outputs = self._activation_function.activate(np.sum(np.dot(inputs, self.get_weights()) + self.get_bias()))
        return self._outputs
        
    def backpropagate(self, error, learning_rate):
        """Adjusts the weights coming into this layer based on the given output error vector.
        
        For the output layer, the "error" vector should be a list of output errors, y_k - t_k.
        For a hidden layer, the "error" vector should be a list of the delta values from the following layer, such as delta_z_k
        
        Returns a list of the delta values for each node in this layer. These deltas can be used as the error
        values when calling backpropagate on the previous layer."""
        #assert isinstance(error, np.ndarray)
        #assert isinstance(self._prev_layer._outputs, np.ndarray)
        #assert isinstance(self._outputs, np.ndarray)  

        # Compute deltas. 
        deltas = np.dot(error, self._activation_function.derivative_given_y(self._outputs))
        
        # Compute gradient.
        synapse = np.dot(np.dot(error, self._prev_layer._outputs), learning_rate)
       
        # Adjust weights.
        self._weights = np.add(self._weights, synapse)
        
        # Adjust bias weights.
        if self._use_bias:
            self._bias = np.add(self._bias, synapse)
            
        return deltas
        
X_data = np.array([[0,0],[1,0],[0,1],[1,1]])
y_data = np.array([[0,1,1,0]]).T
print(X_data)
print(y_data)

nnet = NNet()
nnet.add_input_layer(2)
nnet.add_dense_layer(2, weight_initializer=WeightInitializer, activation_function=SigmoidActivation)
nnet.add_dense_layer(1, weight_initializer=WeightInitializer, activation_function=SigmoidActivation, name='Output')
nnet.summary()

print("Single")
nnet.train_single_example(X_data, y_data, learning_rate=0.5)
nnet.summary(verbose=True)
print("Full")
nnet.train(X_data, y_data, learning_rate=0.5, num_epochs=10)
nnet.summary(verbose=True)

out = nnet.predict(X_data[0])
print("predicted:%s" % (out))