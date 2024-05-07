import math
import random
import numpy as np

    
# Data point is one part of the data that can be completely fed to the neural
# network to get one output.
class data_point:
    
    def __init__(self, inputs, outputs) -> None:
        self.inputs: np.ndarray = inputs
        self.expected_outputs: np.ndarray = outputs


# This class stores the calculated data for each layer in order to be used in the
# calculation of the values for other layers.
class layer_learn_data:

    def __init__(self, layer) -> None:
        self.inputs: np.ndarray = np.array([])
        self.weighted_inputs: np.ndarray = np.zeros(layer.num_nodes_out)
        self.activations: np.ndarray = np.zeros(layer.num_nodes_out)
        self.node_values: np.ndarray = np.zeros(layer.num_nodes_out)


class layer:

    # Creating the layer
    def __init__(self, num_nodes_in: int, num_nodes_out: int) -> None:
        
        self.num_nodes_in: int = num_nodes_in
        self.num_nodes_out: int = num_nodes_out

        self.weights: np.ndarray = np.zeros(num_nodes_in * num_nodes_out)
        self.biases: np.ndarray = np.zeros(num_nodes_out)

        self.cost_gradient_w: np.ndarray = np.zeros(len(self.weights))
        self.cost_gradient_b: np.ndarray = np.zeros(len(self.biases))

        
        self.initialize_random_weights()

    
    # Calculate the output of the layer
    def calculate_output(self, inputs: np.ndarray) -> np.ndarray:
        activations: np.ndarray = np.zeros(self.num_nodes_out)

        for node_out in range(0, self.num_nodes_out):
            weighted_input: float = self.biases[node_out]
            for node_in in range(0, self.num_nodes_in):
                weighted_input += inputs[node_in] * self.weights[self.get_flat_index(node_in, node_out)]
            
            activations[node_out] = self.activation_function(weighted_input)

        return activations


    # Calculate layer output activations and store inputs/weightedInputs/activations in the given learnData object
    def calculate_outputs(self, inputs: np.ndarray, learn_data: layer_learn_data) -> np.ndarray:
        learn_data.inputs = inputs

        for node_out in range(0, self.num_nodes_out):
            weighted_input = self.biases[node_out]
            
            for node_in in range(0, self.num_nodes_in):
                weighted_input += inputs[node_in] * self.weights[self.get_flat_index(node_in, node_out)]
            learn_data.weighted_inputs[node_out] = weighted_input

            # Applying the activation function
            learn_data.activations[node_out] = self.activation_function(weighted_input)

        return learn_data.activations


    # Update the weights and biases based on the cost gradients (gradient descent)
    def apply_gradients(self, learn_rate):
        for node_out in range(0, self.num_nodes_out):
            self.biases[node_out] -= self.cost_gradient_b[node_out] * learn_rate
            for node_in in range(0, self.num_nodes_in):
                self.weights[self.get_flat_index(node_in, node_out)] -= self.cost_gradient_w[self.get_flat_index(node_in, node_out)] * learn_rate


    # Calculate the "node values" for the output layer. This is an array containing for each node:
    # the partial derivative of the cost with respect to the weighted input
    def calculate_output_layer_node_values(self, layer_learn_data: layer_learn_data, expected_outputs: np.ndarray):
        node_values: np.ndarray = np.zeros(len(expected_outputs))

        for i in range(0, len(node_values)):
            # Evaluating partial derivatives for current node: cost/activation & activation/weighted_input
            # to minimize the cost function which calculates the difference between the expected output and
            # the current output.
            cost_derivative: float = self.node_cost_derivative(layer_learn_data.activations[i], expected_outputs[i])
            activation_derivative: float = self.activation_derivative(layer_learn_data.weighted_inputs[i])
            node_values[i] = cost_derivative * activation_derivative

        layer_learn_data.node_values = node_values

        return node_values
    
    # Calculate the "node values" for a hidden layer. This is an array containing for each node:
	# the partial derivative of the cost with respect to the weighted input
    def calculate_hidden_layer_node_values(self, layer_learn_data: layer_learn_data, old_layer, old_node_values: np.ndarray) -> np.ndarray:

        for new_node_index in range(0, self.num_nodes_out):
            new_node_value: float = 0.0
            for old_node_index in range(0, len(old_node_values)):
                # Partial derivative of the weighted input with respect to the input
                weighted_input_derivative = old_layer.weights[old_layer.get_flat_index(new_node_index, old_node_index)]
                new_node_value += weighted_input_derivative * old_node_values[old_node_index]
            new_node_value *= self.activation_derivative(layer_learn_data.weighted_inputs[new_node_index])
            layer_learn_data.node_values[new_node_index] = new_node_value

        return layer_learn_data.node_values


    def update_gradients(self, layer_learn_data: layer_learn_data) -> None:
        for node_out in range(0, self.num_nodes_out):
            node_value: float = layer_learn_data.node_values[node_out]
            for node_in in range(0, self.num_nodes_in):
                # Evaluate the partial derivative: cost / weight of current connection
                derivative_cost_wrt_weight: float = layer_learn_data.inputs[node_in] * node_value
                # The cost_gradient_w array stores these partial derivatives for each weight.
                self.cost_gradient_w[self.get_flat_index(node_in, node_out)] += derivative_cost_wrt_weight

            # Evaluate the partial derivative: cost / bias of the current node
            derivate_cost_wrt_bias = 1 * node_value
            self.cost_gradient_b[node_out] += derivate_cost_wrt_bias
        return None


    # Returns the index of the weight from `node_in` to `node_out`
    def get_flat_index(self, node_in, node_out) -> int:
        flat_index: int = node_out * self.num_nodes_in + node_in
        return flat_index
    
    # An activation function helps with more complex decision boundary
    def activation_function(self, weighted_input: float) -> float:
        return 1 / (1 + math.exp(-weighted_input)) # Sigmoid function


    # Calculate the derivative of the activation function to speed up the cost calculation
    def activation_derivative(self, weighted_input: float) -> float:
        activation: float = self.activation_function(weighted_input)
        return activation * (1 - activation)
    

    # Measuring the error on each node's value
    def node_cost(self, output_activation: float, expected_output: float) -> float:
        error: float = output_activation - expected_output
        return error ** 2
    

     # Partial derivative of the cost with respect to the activation of an output node
    def node_cost_derivative(self, output_activation: float, expected_output: float) -> float:
        return 2 * (output_activation - expected_output)
    

    # Initialize the weights of the network to random values
    def initialize_random_weights(self) -> None:
        for node_in in range(0, self.num_nodes_in):
            for node_out in range(0, self.num_nodes_out):
                # Get a random value between -1 and +1
                random_value: float = random.random() * 2 - 1
                # Scale the random value by 1 / sqrt(num_inputs)
                self.weights[self.get_flat_index(node_in, node_out)] = random_value / math.sqrt(self.num_nodes_in)
        return None
