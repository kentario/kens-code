from helper import directions, contains, remove, Rewards

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
        # self.directions[-1] is the current direction.
        self.directions = deque([directions[1] for _ in range(5)], maxlen=5)

    # None can be used to represent no turn.
    def turn (self, direction):
        if (direction is None):
            direction = self.directions[-1]
            
        self.directions.append(direction)

    # Returns true if an apple was eaten, false if not.
    def move (self, apples):
        self.body = np.vstack([self.head, self.body])
        self.head += self.directions[-1]

        # Check if the head is in an apple.
        if (contains(apples, self.head)):
            self.length += 1
            return True
        else:
            # Only shorten the body if the head is not in an apple.
            self.body = self.body[:-1]
            return False

class Game:
    def __init__ (self, width: int, height: int, num_apples: int):
        self.width = width
        self.height = height
        self.num_apples = num_apples
        self.num_apples_current = num_apples
        self.snake = Snake()
        self.apple_locations = np.empty((num_apples, 2), dtype=int)
        self.num_ticks = 0

        self.observation_size = 23
        self.action_size = 4

        for i in range(num_apples):
            self.apple_locations[i] = self.new_apple_location()

    def reset (self):
        self.num_apples_current = self.num_apples
        self.snake = Snake()
        self.apple_locations = np.empty((self.num_apples, 2), dtype=int)
        self.num_ticks = 0
        
        for i in range(self.num_apples):
            self.apple_locations[i] = self.new_apple_location()
            
    def win (self):
        return self.snake.length >= self.width * self.height

    def lose (self):
        # The game is lost if either:
        # The snake's head isthe screen.
        return ((self.snake.head[0] < 0 or self.snake.head[0] >= self.width or
                 self.snake.head[1] < 0 or self.snake.head[1] >= self.height) or
        # Or if the snake's head is in its body.
                (contains(self.snake.body, self.snake.head)))

    def new_apple_location (self):
        max_tries = 1000
        for i in range(max_tries):
            apple_location = np.array([random.randint(0, self.width - 1),
                                          random.randint(0, self.height - 1)])

            if (not contains(self.snake.body, apple_location) and
                not np.array_equal(apple_location, self.snake.head) and
                not contains(self.apple_locations, apple_location)):
                return apple_location

        raise ValueError("No valid apple location found")

    # Returns the reward:
    # -1 for losing, +1 for eating an apple, +10 for winning.
    def update (self, turn=None):
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
                
    def get_state (self):
        # 3 is apple, 2 is snake, 1 is wall, 0 is empty.
        surroundings = np.empty(8, dtype=int)
        for i, direction in enumerate([(-1, -1), (0, -1), (1, -1),
                                       (-1,  0),          (1,  0),
                                       (-1,  1), (0,  1), (1,  1)]):
            l = np.array([self.snake.head[0] + direction[0], self.snake.head[1] + direction[1]])
            if (l[0] < 0 or l[0] >= self.width or l[1] < 0 or l[1] >= self.height):
                # Wall.
                surroundings[i] = 1
            elif (contains(self.snake.body, l)):
                # Snake.
                surroundings[i] = 2
            elif (contains(self.apple_locations, l)):
                # Apple.
                surroundings[i] = 3
            else:
                # Empty.
                surroundings[i] = 0

# for direction in past_directions: For each tuple in the past_directions deque
#     for coord in direction: for each part of the direction tuple, first x, then y
#         Add the coordinate
        directions = [coordinate
                      for direction in self.snake.directions
                          for coordinate in direction]

        state = torch.tensor([self.snake.head[0],
                              self.snake.head[1],
                              # Past directions
                              *directions,
                              self.snake.length,
                              self.apple_locations[0][0],
                              self.apple_locations[0][1],
                              # State of squares around snake head.
                              *surroundings])
        return state.float()
