from helper import directions, Rewards

import torch
from torch import nn

class Model(nn.Module):
    def __init__ (self, input_size: int, hidden_sizes: int, output_size: int):
        super().__init__()

        self.l1 = nn.Linear(input_size, hidden_sizes)
        self.l2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.l3 = nn.Linear(hidden_sizes, output_size)

    def forward (self, x):
        x = torch.celu(self.l1(x))
        x = torch.celu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x

class Agent:
    def __init__ (self):
        # Currently it's inputs are:
        # The position of head (2)
        # The last 5 turns/directions including the current direction (10)
        # The length of the snake (1)
        # Position of an apple (2)
        # What is in all directions of the head including diagonal (8)
        # A total of 23 inputs.
        self.model = Model(23, 512, 5)

    # Returns the output of the model given a game.
    def get_output (self, game):
        return self.model(game.get_state())

    # Returns a direction (not index of directions) given a game.
    def get_direction (self, game):
        return directions[torch.argmax(self.get_output(game)).item()]
