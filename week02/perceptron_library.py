import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Perceptron:
    def __init__(self, weights, bias, binary=True):
        self.weights = np.array(weights, dtype=float)
        self.bias = bias
        self.binary = binary
        self.epochs = -1

    def __str__(self):
        return f"Perceptron(weights={self.weights}, bias={self.bias}, binary={self.binary})"

    def activation(self, x):
        if self.binary:
            return np.where(x > 0, 1, np.where(x < 0, 0, 0.5))
        else:
            return np.where(x > 0, 1, np.where(x < 0, -1, 0))

    def forward(self, inputs):
        potential = np.dot(inputs, self.weights) + self.bias  #Corrected dot product
        return self.activation(potential)

    def rosenblatt_batch(self, training_inputs, true_outputs, learning_rate = 1, epochs=100, print_progress=True):
        if epochs > 1:
            if print_progress: print("Rosenblatt batch:")
        else:
            if print_progress: print("Hebbian")
        for epoch in range(epochs):
            predictions = self.forward(training_inputs)
            errors = sign(true_outputs - predictions)

            # Batch update
            # self.weights += np.sum(errors * training_inputs, axis=0)
            self.weights += learning_rate * training_inputs.T @ errors # more efficient: transposition and matrix multiplication
            self.bias += learning_rate * np.sum(errors)
            if print_progress:
                print(f"Epoch: {epoch} Weights: {self.weights} Bias: {self.bias} {np.sum(np.abs(errors))}")
            if np.sum(np.abs(errors)) == 0:
                break
        if print_progress:
            print(f"The training ended after {epoch+1} epochs.")
        self.epochs = epoch+1

    def hebbian(self, training_inputs, true_outputs, print_progress=True):
        return self.rosenblatt_batch(training_inputs, true_outputs, 1, 1, print_progress)

    def rosenblatt_iterative(self, training_inputs, true_outputs, learning_rate = 1, epochs=100, print_progress=True):
        if print_progress:
            print("Rosenblatt iterative:")
        for epoch in range(epochs):

            # Shuffle the training data for each epoch
            shuffled_indices = np.random.permutation(len(training_inputs))
            training_inputs = training_inputs[shuffled_indices]
            true_outputs = true_outputs[shuffled_indices]

            errors = 0
            for inputs, true_output in zip(training_inputs, true_outputs):
                prediction = self.forward(inputs)
                error = sign(true_output - prediction)
                if abs(error) and print_progress:
                    print(f"Epoch: {epoch}, Inputs: {inputs}, Prediction: {prediction} Old weights: {self.weights} {self.bias} New weights: {self.weights+learning_rate*error*inputs} {self.bias+learning_rate*error}")
                self.weights += learning_rate * error * inputs
                self.bias += learning_rate * error
                errors += abs(error) # Accumulate the absolute errors
            if errors == 0: # Check if there were any errors in this epoch.
                break # End training if no errors were found
        if print_progress:
            print(f"The training ended after {epoch+1} epochs.")
        self.epochs = epoch+1

    def rosenblatt_iterative_best(self, training_inputs, true_outputs, learning_rate = 1, epochs=100, print_progress=True):
        if print_progress:
            print("Rosenblatt iterative + store best solution:")
        min_errors = float('inf')
        best_epoch = -1
        for epoch in range(epochs):

            # Shuffle the training data for each epoch
            shuffled_indices = np.random.permutation(len(training_inputs))
            training_inputs = training_inputs[shuffled_indices]
            true_outputs = true_outputs[shuffled_indices]

            errors = 0
            for inputs, true_output in zip(training_inputs, true_outputs):
                prediction = self.forward(inputs)
                error = sign(true_output - prediction)
                if abs(error) and print_progress:
                    print(f"Epoch: {epoch}, Inputs: {inputs}, Prediction: {prediction} Old weights: {self.weights} {self.bias} New weights: {self.weights+learning_rate*error*inputs} {self.bias+learning_rate*error}")
                self.weights += learning_rate * error * inputs
                self.bias += learning_rate * error
                errors += abs(error) # Accumulate the absolute errors
            perceptr_error = perceptron_error(self.forward(training_inputs), true_outputs)
            if perceptr_error < min_errors:
                min_errors = perceptr_error
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
                best_epoch = epoch
            if errors == 0: # Check if there were any errors in this epoch.
                break # End training if no errors were found
        self.weights = best_weights.copy()
        self.bias = best_bias.copy()
        if print_progress:
            print(f"The best solution found in {best_epoch+1} epochs with error {min_errors}.")
            print(f"The training ended after {epoch+1} epochs.")
        self.epochs = epoch+1
        self.best_epoch = best_epoch



def plot_decision_boundary_2D(perceptron, training_inputs, true_outputs):
    """Plots the decision boundary of the perceptron."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract the first two columns of training_inputs
    x1 = training_inputs[:, 0]
    x2 = training_inputs[:, 1]

    # Generate points for visualization
    x_min, x_max = np.min(x1) - 1, np.max(x1) + 1  # Extend the range slightly
    y_min, y_max = np.min(x2) - 1, np.max(x2) + 1  # Extend the range slightly

    x = np.linspace(x_min, x_max, 100)
    y = -(perceptron.weights[0] * x + perceptron.bias) / perceptron.weights[1]

    # Plot the decision boundary
    plt.plot(x, y, label='Decision Boundary')

    # Plot the points
    for i, input_vector in enumerate(training_inputs):
        if true_outputs[i] == 1:
            plt.scatter(input_vector[0], input_vector[1], color='green', label='Class 1' if i == 0 else "")  # Add label only for the first point of each class
        else:
            plt.scatter(input_vector[0], input_vector[1], color='red', label='Class -1' if i == 0 else "")

    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title('Perceptron Decision Boundary', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)  # Set y-axis limits

    # Highlight x and y axes
    plt.axhline(0, color='black', linewidth=0.8)  # x-axis
    plt.axvline(0, color='black', linewidth=0.8)  # y-axis
    plt.show()

def plot_decision_boundary_3D(perceptron, training_inputs, true_outputs):
    """Plots the decision boundary of the perceptron in 3D."""
    # Select first three columns if more than three exist
    training_inputs = training_inputs[:, :3]

    # Check if the input data has 3 features
    if training_inputs.shape[1] != 3:
        raise ValueError("Input data must have at least 3 features for 3D visualization.")

    # Create a meshgrid of points
    x_min, x_max = training_inputs[:, 0].min() - 1, training_inputs[:, 0].max() + 1
    y_min, y_max = training_inputs[:, 1].min() - 1, training_inputs[:, 1].max() + 1
    z_min, z_max = training_inputs[:, 2].min() - 1, training_inputs[:, 2].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                          np.arange(y_min, y_max, 0.1))

    # Calculate z values for the decision boundary
    zz = (-perceptron.weights[0] * xx - perceptron.weights[1] * yy - perceptron.bias) / perceptron.weights[2]

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the decision boundary
    ax.plot_surface(xx, yy, zz, alpha=0.5)

    # Plot the training data points
    ax.scatter(training_inputs[:, 0], training_inputs[:, 1], training_inputs[:, 2], c=true_outputs, cmap=plt.cm.Paired)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('Perceptron Decision Boundary (3D)')

    plt.show()

def sign(x):
    return np.where(x > 0, 1, np.where(x < 0, -1, 0))

def perceptron_error(true_outputs, predicted_outputs):
    return np.sum(true_outputs != predicted_outputs)
    #return np.mean(true_outputs != predicted_outputs)
  
