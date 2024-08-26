from helper import directions, contains, remove, Rewards, clamp

from collections import deque
import random
import numpy as np
import torch

class Snake:
    def __init__ (self):
        self.length = 2

        self.head = np.array([1, 0])
        self.body = np.array([[0, 0]])

        # Default direction to the right
        self.direction = directions[1]

    # None can be used to represent no turn.
    def turn (self, direction: tuple[int, int]) -> None:
        self.direction = self.direction if direction is None else direction

    # Returns true if an apple was eaten, false if not.
    def move (self, apples: np.array) -> bool:
        self.body = np.vstack([self.head, self.body])
        self.head += self.direction

        # Check if the head is in an apple.
        if (contains(apples, self.head)):
            self.length += 1
            return True
        else:
            # Only shorten the body if the head is not in an apple.
            self.body = self.body[:-1]
            return False

    def get_observation (self) -> torch.tensor:
        state = torch.tensor([
            self.length,
            directions[self.head]
        ], dtype = float)

class Game:
    def __init__ (self, width: int, height: int, num_apples: int):
        self.width = width
        self.height = height
        
        self.num_ticks = 0
        self.snake = Snake()
        self.observation_size = 23
        self.action_size = 4

        self.num_apples = clamp(num_apples, width * height - self.snake.length, 0)
        self.num_apples_current = self.num_apples
        self.apple_locations = np.empty((num_apples, 2), dtype=int)
        self.all_locations = np.array([(x, y) for x in range(width)
                                                  for y in range(height)])
        
        for i in range(self.num_apples):
            self.apple_locations[i] = self.new_apple_location()

    def reset (self) -> None:
        self.num_apples_current = self.num_apples
        self.snake = Snake()
        self.apple_locations = np.empty((self.num_apples, 2), dtype=int)
        self.num_ticks = 0
        
        for i in range(self.num_apples):
            self.apple_locations[i] = self.new_apple_location()
            
    def win (self) -> bool:
        return self.snake.length >= self.width * self.height

    def lose (self) -> bool:
        # The game is lost if either:
        # The snake's head isthe screen.
        return ((self.snake.head[0] < 0 or self.snake.head[0] >= self.width or
                 self.snake.head[1] < 0 or self.snake.head[1] >= self.height) or
        # Or if the snake's head is in its body.
                (contains(self.snake.body, self.snake.head)))

    def new_apple_location (self) -> tuple[int, int]:
        np.random.shuffle(self.all_locations)

        for location in self.all_locations:
            if (not contains(self.snake.body, location) and
                not np.array_equal(location, self.snake.head) and
                not contains(self.apple_locations, location)):
                return location

        raise ValueError("No valid apple location found")

    # Returns the reward:
    # -1 for losing, +1 for eating an apple, +10 for winning.
    def update (self, turn=None) -> Rewards:
        self.num_ticks += 1
        # Turn the snake.
        self.snake.turn(turn)
        # Move the snake.
        apple_eaten = self.snake.move(self.apple_locations)

        # Create new apple if needed.
        if (apple_eaten):
            self.apple_locations = remove(self.apple_locations, self.snake.head)
            # If there is no space for a new apple, don't create a new one.
            if (self.num_apples_current + self.snake.length > self.width * self.height):
                self.num_apples_current -= 1
                # If there are 0 apples, then the game will be won.
                if (self.num_apples_current <= 0):
                    return Rewards.WIN
            else:
                # If there is space for an apple, make one.
                self.apple_locations = np.vstack([self.apple_locations, self.new_apple_location()])
                return Rewards.APPLE

        return Rewards.LOSE if self.lose() else Rewards.SURVIVE

    def get_state (self) -> torch.tensor:
        state = np.zeros((self.width, self.height), dtype = float)

        # 3 is snake head, 2 is snake body, 1 is apple, 0 is empty
        # All the apples
        for apple_location in self.apple_locations:
            state[apple_location[0]][apple_location[1]] = 1
        # The snake body
        for body in self.snake.body:
            state[body[0]][body[1]] = 2
        # The snake head
        state[self.snake.head[0]][self.snake.head[1]] = 3
        
        return torch.from_numpy(state)
