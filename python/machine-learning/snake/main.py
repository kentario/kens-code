import helper as h
import game as g
import learning as l

import pygame

def main ():
    pygame.init()

    width = 10
    height = 10
    pixel_size = 40
    num_apples = 1
    surface = pygame.display.set_mode((width * pixel_size, height * pixel_size))

    game = g.Game(width, height, num_apples)

    num_epochs = 1000

    agent = l.Agent()
    
    h.play_game(game, surface, pixel_size)
    
    pygame.quit()

if __name__ == "__main__":
    main()
