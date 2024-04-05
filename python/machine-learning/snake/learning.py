from helper import directions
from helper import Rewards

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

class Snake_AI:
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

class PPO:
    def __init__ (self, game):
        self._init_hyperparameters()
        
        self.game = game
        self.observation_size = game.observation_size
        self.action_size = game.action_size

        self.actor = Model(observation_size, 256, action_size)
        self.critic = Model(observation_size, 1)

    def _init_hyperparameters (self):
        self.steps_per_batch = 4800
        self.max_steps_per_episode = 1600

    # Collect batch data.
    def rollout (self):
        batch_obsservations = []
        batch_actions = []
        batch_log_probabilities = []
        batch_rewards = []
        batch_rewards_to_go = []
        batch_episodic_lengths = []

    def learn (self):
        for t in range(num_epochs):
            pass
