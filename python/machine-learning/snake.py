import random
from collections import deque
import numpy
import pygame

import torch
import torch.nn as nn
import torch.nn.functional as functional

# up, right, down, left, straight.
directions = ((0, -1), (1, 0), (0, 1), (-1, 0), None)

# Returns whether numpy array sub is contained in numpy array arr.
def contains (arr, sub):
    return numpy.any(numpy.all(arr == sub, axis=1))

# Removes a specific 1d numpy array from 2d numpy array arr.
def remove (arr, element):
    return numpy.delete(arr, numpy.where(numpy.all(arr == element, axis=1)), axis=0)

class Snake:
    def __init__ (self):
        self.length = 2

        self.head = numpy.array([1, 0])
        self.body = numpy.array([[0, 0]])

        # Default direction to the right
        # self.directions[-1] is the current direction.
        self.directions = deque([directions[1] for _ in range(5)], maxlen=5)

    # None can be used to represent no turn.
    def turn (self, direction):
        # If the components add to 0, then it is a 180 degree turn, which is the same as no turn.
        if (direction is None or
            self.directions[-1][0] + direction[0] == 0 or
            self.directions[-1][1] + direction[1] == 0):
            direction = self.directions[-1]
            
        self.directions.append(direction)

    # Returns true if an apple was eaten, false if not.
    def move (self, apples):
        self.body = numpy.vstack([self.head, self.body])
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
        # +1 for surviving, +2 for eating an apple.
        self.snake = Snake()
        self.apple_locations = numpy.empty((num_apples, 2), dtype=int)

        for i in range(num_apples):
            self.apple_locations[i] = self.new_apple_location()

    def win (self):
        return self.snake.length >= self.width * self.height

    def lose (self):
        # The game is lost if either:
        # The snake's head is outside the screen.
        return ((self.snake.head[0] < 0 or self.snake.head[0] >= self.width or
                 self.snake.head[1] < 0 or self.snake.head[1] >= self.height) or
        # Or if the snake's head is in its body.
                (contains(self.snake.body, self.snake.head)))

    def new_apple_location (self):
        max_tries = 1000
        for i in range(max_tries):
            apple_location = numpy.array([random.randint(0, self.width - 1),
                                          random.randint(0, self.height - 1)])

            if (not contains(self.snake.body, apple_location) and
                not numpy.array_equal(apple_location, self.snake.head) and
                not contains(self.apple_locations, apple_location)):
                return apple_location

        raise ValueError("No valid apple location found")

    # Win loss detection is done from outside this method.
    def update (self, turn=None):
        # Turn the snake.
        self.snake.turn(turn)
        # Move the snake.
        apple_eaten = self.snake.move(self.apple_locations)

        # Create new apple if needed.
        if (apple_eaten):
            self.apple_locations = remove(self.apple_locations, self.snake.head)
            # If there is no space for a new apple, don't create a new one.
            if (self.num_apples + self.snake.length > self.width * self.height):
                self.num_apples -= 1
            else:
                # If there is space for an apple, make one.
                self.apple_locations = numpy.vstack([self.apple_locations, self.new_apple_location()])

class Model(nn.Module):
    def __init__ (self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward (self, x):
        x = torch.celu(self.l1(x))
        x = torch.sigmoid(self.l2(x))
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

    def get_state (self, game):
        # 3 is apple, 2 is snake, 1 is wall, 0 is empty.
        surroundings = numpy.empty(8, dtype=int)
        for i, direction in enumerate([(-1, -1), (0, -1), (1, -1),
                                       (-1,  0),          (1,  0),
                                       (-1,  1), (0,  1), (1,  1)]):
            l = numpy.array([game.snake.head[0] + direction[0], game.snake.head[1] + direction[1]])
            if (l[0] < 0 or l[0] >= game.width or l[1] < 0 or l[1] >= game.height):
                # Wall.
                surroundings[i] = 1
            elif (contains(game.snake.body, l)):
                # Snake.
                surroundings[i] = 2
            elif (contains(game.apple_locations, l)):
                # Apple.
                surroundings[i] = 3
            else:
                # Empty.
                surroundings[i] = 0

# for direction in past_directions: For each tuple in the past_directions deque
#     for coord in direction: for each part of the direction tuple, first x, then y
#         Add the coordinate
        directions = [coordinate
                      for direction in game.snake.directions
                          for coordinate in direction]

        state = torch.tensor([game.snake.head[0],
                              game.snake.head[1],
                              # Past directions
                              *directions,
                              game.snake.length,
                              game.apple_locations[0][0],
                              game.apple_locations[0][1],
                              # State of squares around snake head.
                              *surroundings])
        return state.float()

    # Returns the output of the model given a game.
    def get_output (self, game):
        return self.model(self.get_state(game))

    # Returns a direction (not index of directions) given a game.
    def get_direction (self, game):
        return directions[torch.argmax(self.get_output(game)).item()]

def play_game (game, surface, pixel_size):
    clock = pygame.time.Clock()
    fps = 3
    running = True
    while (running):
        turn = None
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False
                break;
            # Check if the user wants to turn the snake.
            elif (event.type == pygame.KEYDOWN):
                if (event.key in [pygame.K_w, pygame.K_UP]):
                    turn = directions[0]
                    break
                elif (event.key in [pygame.K_a, pygame.K_LEFT]):
                    turn = directions[3]
                    break
                elif (event.key in [pygame.K_s, pygame.K_DOWN]):
                    turn = directions[2]
                    break
                elif (event.key in [pygame.K_d, pygame.K_RIGHT]):
                    turn = directions[1]
                    break

        game.update(turn)
                
        if (game.win()):
            print("Win")
            running = False
        if (game.lose()):
            print("Lose")
            running = False

        surface.fill("black")
        # Draw the snake
        for s in numpy.vstack([game.snake.head, game.snake.body]):
            rect = pygame.Rect((s[0] * pixel_size, s[1] * pixel_size), (pixel_size, pixel_size))
            pygame.draw.rect(surface, (0, 255, 0), rect)
        # Draw the apples.
        for a in game.apple_locations:
            rect = pygame.Rect((a[0] * pixel_size, a[1] * pixel_size), (pixel_size, pixel_size))
            pygame.draw.rect(surface, (255, 0, 0), rect)
                    
        pygame.display.flip()
                    
        delta = clock.tick(fps)
    
def main ():
    pygame.init()

    width = 20
    height = 20
    pixel_size = 20
    surface = pygame.display.set_mode((width * pixel_size, height * pixel_size))

    game = Game(width, height, 1)    

    num_epochs = 1000

    snake_ai = Snake_AI()

    print(snake_ai.get_direction(game))
    
    play_game(game, surface, pixel_size)
    
    pygame.quit()

if __name__ == "__main__":
    main()
