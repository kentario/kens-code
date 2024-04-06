from enum import Enum
from collections import deque
import numpy as np
import pygame

class Rewards(Enum):
    WIN = 10
    APPLE = 1
    SURVIVE = 0
    LOSE = -1

# up, right, down, left, straight.
directions = ((0, -1), (1, 0), (0, 1), (-1, 0), None)

# Returns whether numpy array sub is contained in numpy array arr.
def contains (arr, sub):
    return np.any(np.all(arr == sub, axis=1))

# Removes a specific 1d numpy array from 2d numpy array arr.
def remove (arr, element):
    return np.delete(arr, np.where(np.all(arr == element, axis=1)), axis=0)

def draw_game (game, surface, pixel_size, fancy_snake = True):
    border_radius = int(pixel_size/10)
    
    surface.fill("black")
    # Draw the snake body.
    for i, body in enumerate(game.snake.body):
        # Todo: Find way to scale i based on board width and height.
        i = i if fancy_snake else 0
        rect = pygame.Rect(
            (body[0] * pixel_size, body[1] * pixel_size),
            (pixel_size, pixel_size))
        pygame.draw.rect(surface,
                         (i*3, 255-i*2, i*3),
                         rect,
                         border_radius=border_radius+i)

    # Draw the snake head.
    head = game.snake.head
    pygame.draw.rect(surface,
                     (255, 165, 0),
                     pygame.Rect(
                         (head[0] * pixel_size, head[1] * pixel_size),
                         (pixel_size, pixel_size)),
                     border_radius=border_radius)
    
    # Draw the apples.
    for apple in game.apple_locations:
        rect = pygame.Rect((apple[0] * pixel_size, apple[1] * pixel_size), (pixel_size, pixel_size))
        pygame.draw.rect(surface,
                         (255, 0, 0),
                         rect,
                         border_radius=border_radius * 2)
        
    pygame.display.flip()

def play_game (game, surface, pixel_size, agent=None):
    input_buffer = deque()
    
    clock = pygame.time.Clock()
    fps = 5
    running = True
    turn = None
    while (running):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False
                break;
            # Check if the user wants to turn the snake.
            if (agent is None):
                if (event.type == pygame.KEYDOWN):
                    if (event.key == pygame.K_r):
                        game.reset()
                    if (event.key in [pygame.K_w, pygame.K_UP]):
                        input_buffer.append(directions[0])
                    elif (event.key in [pygame.K_a, pygame.K_LEFT]):
                        input_buffer.append(directions[3])
                    elif (event.key in [pygame.K_s, pygame.K_DOWN]):
                        input_buffer.append(directions[2])
                    elif (event.key in [pygame.K_d, pygame.K_RIGHT]):
                        input_buffer.append(directions[1])
            else:
                input_buffer.append(agent.get_direction(game))
                
        turn = None if len(input_buffer) <= 0 else input_buffer.popleft()
        result = game.update(turn)

        if (result is Rewards.WIN):
            print("win")
            running = False
        if (result is Rewards.LOSE):
            print("Lose")
            running = False

        draw_game(game, surface, pixel_size)
            
        delta = clock.tick(fps)
