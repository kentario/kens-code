import helper as h
import game as g
import learning as l

import pygame

def main ():
    pygame.init()

    # Parameters for the game.
    width = 10
    height = 10
    pixel_size = 40
    num_apples = 1
    surface = pygame.display.set_mode((width * pixel_size, height * pixel_size))

    game = g.Game(width, height, num_apples)
    agent = l.Agent()

    # Hyperparameters
    num_epochs = 1000

    for epoch in range(num_epochs):
        game.reset()
        state = game.get_state()

        while (not game.lose() and 
        
    pygame.quit()

if __name__ == "__main__":
    main()
