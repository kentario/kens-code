from helper import play_game
from game import Game
from learning import Snake_AI

import pygame

def main ():
    pygame.init()

    width = 10
    height = 10
    pixel_size = 40
    surface = pygame.display.set_mode((width * pixel_size, height * pixel_size))

    game = Game(width, height, 1)

    num_epochs = 1000

    agent = Snake_AI()
    
    play_game(game, surface, pixel_size)
    
    pygame.quit()

if __name__ == "__main__":
    main()
