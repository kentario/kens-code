from enum import Enum
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

def play_game (game, surface, pixel_size, agent=None):
    clock = pygame.time.Clock()
    fps = 4
    running = True
    while (running):
        turn = None
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False
                break;
            # Check if the user wants to turn the snake.
            if (agent is None):
                if (event.type == pygame.KEYDOWN):
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
            else:
                turn = agent.get_direction(game)

        result = game.update(turn)

        if (result is Rewards.WIN):
            print("win")
            running = False
        if (result is Rewards.LOSE):
            print("Lose")
            running = False

        surface.fill("black")
        # Draw the snake
        for s in np.vstack([game.snake.head, game.snake.body]):
            rect = pygame.Rect((s[0] * pixel_size, s[1] * pixel_size), (pixel_size, pixel_size))
            pygame.draw.rect(surface, (0, 255, 0), rect)
        # Draw the apples.
        for a in game.apple_locations:
            rect = pygame.Rect((a[0] * pixel_size, a[1] * pixel_size), (pixel_size, pixel_size))
            pygame.draw.rect(surface, (255, 0, 0), rect)
                    
        pygame.display.flip()
                    
        delta = clock.tick(fps)
