from helper import directions, Rewards
from game import Game

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__ (self, input_size: int, hidden_sizes: int, output_size: int):
        super().__init__()

        self.l1 = nn.Linear(input_size, hidden_sizes)
        self.l2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.l3 = nn.Linear(hidden_sizes, output_size)

    def forward (self, x) -> torch.tensor:
        x = F.celu(self.l1(x))
        x = F.celu(self.l2(x))
        x = F.sigmoid(self.l3(x))
        
        return x

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    def save (self, file_path: str) -> None:
        torch.save(self, file_path)

    def load (self, file_path: str) -> nn.Module:
        if (os.path.exists(file_path)):
            self.load_state_dict(torch.load(file_path))
        else:
            raise FileNotFoundError("Model file not found")

        return self

class Trainer:
    def __init__ (self, model: nn.Module, learn_rate: float, gamma: float):
        self.model = model
        self.learn_rate = learn_rate
        self.gamma = gamma

        self.optimizer = optim.adam(model.parameters(), lr=learn_rate)
        self.loss_f = nn.MSELoss()

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
    def get_output (self, game: Game) -> torch.tensor:
        return self.model(game.get_state())

    # Returns a direction (not index of directions) given a game.
    def get_direction (self, game: Game) -> tuple[int, int]:
        return directions[torch.argmax(self.get_output(game)).item()]
