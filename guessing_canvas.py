from my_nn.neural_network import neural_network

import numpy as np
import tkinter as tk
import math

class App(tk.Frame):
    def __init__(self, side_width: int, pixels: int, guessing_network: neural_network, master=None):
        super().__init__(master)
        self.side_width = side_width
        self.pixels = pixels
        self.width_per_pixel = side_width / pixels
        self.guessing_network = guessing_network

        self.coordinates = np.zeros(pixels*pixels)

        self.canvas = tk.Canvas(self, width=side_width, height=side_width, background="black")
        self.canvas.bind('<Button-1>', self.left_click)
        self.canvas.bind('<B1-Motion>', self.left_click)
        self.canvas.pack()

    def left_click(self, event) -> None:
        base_coord_x = math.floor(event.x/self.width_per_pixel)
        base_coord_y = math.floor(event.y/self.width_per_pixel)
        top_left_coord_x = base_coord_x * self.width_per_pixel
        top_left_coord_y = base_coord_y * self.width_per_pixel
        self.canvas.create_rectangle(top_left_coord_x, top_left_coord_y, top_left_coord_x + self.width_per_pixel, top_left_coord_y + self.width_per_pixel, fill="white", outline="")
        if (self.get_flat_index(base_coord_x, base_coord_y)) >= self.pixels * self.pixels:
            return None
        self.coordinates[(self.get_flat_index(base_coord_x, base_coord_y))] = 1
        
        # Guessing the current canvas
        guessed_answer = self.guessing_network.classify(self.coordinates)
        print(f"I guess a {guessed_answer}")
        # Guess with Confidence Values
        # guessed_answer, confidence = self.guessing_network.classify_with_confidence(self.coordinates)
        # print(f"I guess a {guessed_answer}\n with confidence of {confidence * 100:.2}%")
        return None

    # Returns the index of the weight from `node_in` to `node_out`
    def get_flat_index(self, node_in, node_out) -> int:
        flat_index: int = node_out * self.pixels + node_in
        return flat_index