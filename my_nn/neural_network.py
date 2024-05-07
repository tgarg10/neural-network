from my_nn.layer import layer, data_point, layer_learn_data

import numpy as np
import math

class neural_network:

    def __init__(self) -> None:
        self.layers:np.ndarray = np.array([])
        self.network_learn_data: np.ndarray = np.array([])


    # Create the Neural Network
    def neural_network(self, layer_sizes: np.ndarray) -> None:
        layers: np.ndarray = np.array([layer for i in range(len(layer_sizes) - 1)])

        for i in range(len(layers)):
            layers[i] = layer(layer_sizes[i], layer_sizes[i+1])

        self.layers = layers
        self.network_learn_data = self.initializing_network_learn_data(self.layers)


    # Run the inputs through the network and calculate which output node has the highest value.
    def classify(self, inputs: np.ndarray) -> int:
        outputs: np.ndarray = self.calculate_outputs(inputs)
        return self.index_of_max_value(outputs)
    
    
    # Run the inputs through the network and calculate which output node has the highest value
    # and the confidence value. 
    def classify_with_confidence(self, inputs: np.ndarray) -> tuple:
        outputs: np.ndarray = self.calculate_outputs(inputs)
        highest_value_index: int = self.index_of_max_value(outputs)
        return (highest_value_index, outputs[highest_value_index])


    # Run the input values through the network to calculate the output values.
    def calculate_outputs(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.calculate_output(inputs)
        
        return inputs


    # Run a single iteration of gradient descent (using the finte-difference method)
    def learn(self, training_data: np.ndarray, learn_rate: float) -> None:
        # Use the backpropagation algorithm to calculate the gradient of the cost function
        # (with respect to the network's weights and biases). This is done for each data
        # point, and the gradients are added together.
        for data_point in training_data:
            self.update_all_gradients(data_point, self.network_learn_data)

        # Gradient descent step: update all the weights and biases in the network
        self.apply_all_gradients(learn_rate / len(training_data))

        # Reset all gradients to zero to be ready for the next training data.
        self.clear_all_gradients()


    # Run the inputs through the network.         
    def update_all_gradients(self, data_point: data_point, network_learn_data: np.ndarray) -> None:
        
        # Feed data through the network to calculate outputs.
        # Save all inputs/weighted_inputs/activations along the way to use for backpropagation
        inputs_to_next_layer: np.ndarray = data_point.inputs

        for i in range(0, len(self.layers)):
            inputs_to_next_layer = self.layers[i].calculate_outputs(inputs_to_next_layer, network_learn_data[i])

        # Back propagation
        # During this process, each layer will store the values we need, such as the weighted inputs and activations.
        output_layer: layer = self.layers[len(self.layers) - 1]
        node_values: np.ndarray = output_layer.calculate_output_layer_node_values(network_learn_data[len(self.layers) - 1], data_point.expected_outputs)
        output_layer.update_gradients(network_learn_data[len(self.layers) - 1])

        # Loop backwards through all the hidden layers and update their gradients
        for hidden_layer_index in range(len(self.layers) - 2, -1, -1):
            hidden_layer: layer = self.layers[hidden_layer_index]
            node_values = hidden_layer.calculate_hidden_layer_node_values(network_learn_data[hidden_layer_index], self.layers[hidden_layer_index + 1], node_values)
            hidden_layer.update_gradients(network_learn_data[hidden_layer_index])

            
    # Test the network with testing data and return the correct predictions.
    def testing_network(self, testing_data: np.ndarray) -> int:
        correct_predictions: int = 0

        for data_point in testing_data:
            output: int = self.classify(data_point.inputs)
            expected_output: int = self.index_of_max_value(data_point.expected_outputs)
            if output == expected_output:
                correct_predictions += 1
            
        return correct_predictions


    def training_network(self, data_points: np.ndarray, learn_rate: float, batch_size: int, split_ratio: float):
    
        # Splitting the data into training and testing datasets.
        total_batches: int = math.ceil(len(data_points) / batch_size)
        for batch_number in range(0, total_batches):
            batch_starting_index: int = batch_number * batch_size
            if (batch_starting_index + batch_size) <= len(data_points):
                batch_ending_index: int = batch_starting_index + batch_size
            else:
                batch_ending_index: int = len(data_points)
            current_batch: np.ndarray = data_points[batch_starting_index: batch_ending_index]
            splitting_index: int = math.floor(len(current_batch) * split_ratio)
            training_data: np.ndarray = current_batch[0: splitting_index]
            testing_data: np.ndarray = current_batch[splitting_index: ]
            print(f"Running Batch Number {batch_number + 1} (length {len(current_batch)}) ... ")
            self.learn(training_data, learn_rate)
            print(f"{self.testing_network(testing_data)} / {len(testing_data)}")


    # Creating a network out of the learning data for each layer.
    def initializing_network_learn_data(self, layers: np.ndarray) -> np.ndarray:
        layer_data: np.ndarray = np.array([layer_learn_data for i in range(len(layers))])
        for layer_index in range(0, len(layers)):
            layer_data[layer_index] = layer_learn_data(layers[layer_index])

        return layer_data


    # Returns the first index of the maximum value in the list.
    def index_of_max_value(self, input_list: np.ndarray) -> int:
        return input_list.argmax(axis = 0)


    # Loss function: takes in data points to calculate the error in the current neural network. 
    def cost(self, data_point: data_point) -> float:
        outputs: np.ndarray = self.calculate_outputs(data_point.inputs)
        output_layer: layer = self.layers[len(self.layers) - 1]
        cost: float = 0.0

        for node_out in range(0, len(outputs)):
            cost += output_layer.node_cost(outputs[node_out], data_point.expected_outputs[node_out])

        return cost
    

    # Calculates total error over all the data points.
    def data_cost(self, data_points: np.ndarray) -> float:
        total_cost: float = 0.0

        for data_point in data_points:
            total_cost += self.cost(data_point)

        return total_cost / len(data_points)
    

    # Applies gradients to all layers using learn rate.
    def apply_all_gradients(self, learn_rate: float) -> None:
        for layer in self.layers:
            layer.apply_gradients(learn_rate)
    

    def clear_all_gradients(self) -> None:
        for layer in self.layers:
            layer.cost_gradient_w = np.zeros(len(layer.cost_gradient_w))
            layer.cost_gradient_b = np.zeros(len(layer.cost_gradient_b))
