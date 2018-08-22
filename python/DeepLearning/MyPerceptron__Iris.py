import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import pandas as pd

class MyPerceptron():
    """Simple perceptron with 2 inputs and 1 output."""
    
    def __init__(self, learning_rate = 0.01, num_iterations=10):
        # Initialize the perceptron weights and the bias term.
        self._w = [0, 0]
        self._b = [0, 0]
        self._errors = []
        self._num_iterations = num_iterations
        self._learning_rate = learning_rate
    
    def predict(self, x, verbose=False):
        """x is the input weight vector. Output is the result of running the perceptron on this input.
        
        Implement the Perceptron rule that involves multiplying weights by input, adding in bias, using a threshold, etc.
        
        The returned output should be 1 or 0.
        
        Use the "verbose" flag to print debugging info if desired.
        """
        activation = self._w[0]
        for i in range(len(x)-1):
            activation += self._w[i + 1] * x[i]
        
        if verbose:
            # Print computation results here if desired.
            pass

        return 1 if activation >= 0 else 0
        
    def accuracy(self, x, y):
        """Compute the total % accuracy over a set of inputs x and corresponding outputs y."""
        correct = 0
        for i in range(len(x)):
            example_x = x[i]
            example_y = y[i]
            if self.predict(example_x) == example_y:
                correct += 1
        return float(correct) / len(x)
            
    def update_weights(self, x, target, ctr, l_rate=1, verbose=False):
        """Update the perceptron's weights according to the perceptron learning rule.
        
        x is an input example, and target is the desired output.
        
        This function should modify self._b and self._w. It has no return value.
        
        Use the "verbose" flag to print debugging info if desired.
        """
        current_output = self.predict(x)
        error = target - current_output
        self._w[ctr] = self._w[ctr] + l_rate * error
        
    def train(self, x, y, verbose=False):
        """Train the perceptron for the given number of iterations on the input data x with 
        corresponding target values y.
        
        Use the "verbose" flag to print debugging info if desired.
        """
        assert(len(x) == len(y))

        for i in range(self._num_iterations):
            errors = 0
            print('Iter #%d' % i)
            for j in range(len(x)):
                example_x = x[j]
                example_y = y[j]
                update = self._learning_rate * (example_y - self.predict(example_x))
                self._w[1:] += update * example_x
                self._w[0] += update

                self._b[1:] += update * example_x
                self._b[0] += update                
                errors += int(update != 0.0)
                
            self._errors.append(errors)

            print('Weights:', self._w)
            acc = self.accuracy(x, y)
            print('Accuracy: %.3f%%' % (acc * 100))
            print()


def get_shuffled_data(x, y):
    """Convenient function to shuffle data and outputs, to inject some randomness into training."""
    # Create shuffle pattern of indices.
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    
    # Apply suffle pattern to x and y.
    x_shuffled = x[s]
    y_shuffled = y[s]
    return x_shuffled, y_shuffled

# Import the 'iris' dataset.
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use the first two features.
y = iris.target
x_min = min(X[:, 0])
x_max = max(X[:, 0])

# Map data labels to just two categories.
y_two_categories = np.array([0 if i==0 else 1 for i in y])

X_shuffled, y_shuffled = get_shuffled_data(X, y_two_categories)

p = MyPerceptron()
p.train(X_shuffled, y_shuffled)