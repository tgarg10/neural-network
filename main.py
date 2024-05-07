from my_nn.neural_network import neural_network
from my_nn.layer import data_point
from guessing_canvas import App

import tkinter as tk
import csv
import numpy as np
import pickle


# Normalize between 0.0 and 1.0
def normalize_inputs(numbers) -> np.ndarray:
    # Min-max normalization
    max_number: int = max(numbers)
    min_number: int = min(numbers)
    range_numbers: int = max_number - min_number
    normalized_inputs: np.ndarray = np.array(list(map(lambda x: float((x - min_number) / range_numbers), numbers)))
    
    return normalized_inputs


def csv_to_data_points(csv_reader, possible_outputs_num: int) -> np.ndarray:
    _ = next(csv_reader) # Voiding the header
    reader_list = list(csv_reader)
    data_points: np.ndarray = np.array([data_point for i in range(len(reader_list))])

    for row_index in range(len(reader_list)):
        row: np.ndarray = np.array(list(map(int, reader_list[row_index])))
        expected_outputs: np.ndarray = np.zeros(possible_outputs_num)
        expected_outputs[row[0]] = 1
        inputs: np.ndarray = normalize_inputs(row[1:])
        data_points[row_index] = data_point(inputs, expected_outputs)

    return data_points


def loading_data(file_name: str, possible_outputs_num: int) -> np.ndarray:
    with open(file_name, "r") as data_file:
        csv_reader = csv.reader(data_file, delimiter = ",")
        data_points: np.ndarray = csv_to_data_points(csv_reader, possible_outputs_num)

    return data_points


# Saving the current neural network to a local file
def save_network(neural_network: neural_network, saving_file_name: str):
    with open(saving_file_name, "wb") as saving_file:
        pickle.dump(neural_network, saving_file)


# Loading the existing neural network from a local file
def load_network(stored_file_name: str) -> neural_network:
    with open(stored_file_name, "rb") as stored_file:
        saved_neural_network: neural_network = pickle.load(stored_file)
    return saved_neural_network


def train() -> None:
    split_ratio: float = 0.9 # between 0.0 and 1.0
    data_input_size: int = 784
    data_output_size: int = 10
    learn_rate: float = 0.1
    batch_size: int = 1000
    total_epochs: int = 20
    layer_sizes = np.array([data_input_size, 100, data_output_size])
    file_name = r"train.csv"
    saving_file_name = r"trained_network.txt"

    data_points: np.ndarray = loading_data(file_name, data_output_size)

    # Loading up a partially trained network
    try:
        my_network = load_network(saving_file_name)
    except FileNotFoundError:
        my_network: neural_network = neural_network()
        my_network.neural_network(layer_sizes)  

    # Training the neural network
    for epoch_number in range(total_epochs):
        print(f"Running epoch {epoch_number + 1} ...")
        my_network.training_network(data_points, learn_rate, batch_size, split_ratio)
        save_network(my_network, saving_file_name)

def test() -> None:
    side_width = 560
    pixels = 28
    saving_file_name = r"trained_network.txt"
    my_network = load_network(saving_file_name)
    
    root = tk.Tk()
    gui = App(side_width, pixels, my_network, root)
    gui.pack()
    root.mainloop()

if __name__ == "__main__":
    # train()
    test()