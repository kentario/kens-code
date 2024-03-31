import random
import pygame

#               up     right    down    left  straight
directions = ((0, -1), (1, 0), (0, 1), (-1, 0), None)

class Snake:
    def __init__ (self):
        self.length = 2

        self.head = [1, 0]
        self.body = [[0, 0]]

        # Default direction to the right
        self.direction = directions[1]

    # None can be used to represent no turn.
    def turn (self, direction):
        # If the components add to 0, then it is a 180 degree turn, which is the same as no turn.
        if ((direction is None) or
            (self.direction[0] + direction[0] == 0 and
             self.direction[1] + direction[1] == 0)):
            return
        
        self.direction = direction

    # Returns true if an apple was eaten, false if not.
    def move (self, apples):
        self.body = [self.head] + self.body
        self.head = [self.head[0] + self.direction[0], self.head[1] + self.direction[1]]

        # Check if the head is in an apple.
        if (self.head in apples):
            self.length += 1
            return True
        else:
            # Only shorten the body if the head is not in an apple.
            self.body.pop()
            return False

class Game:
    def __init__ (self, width: int, height: int, num_apples: int):
        self.width = width
        self.height = height
        self.num_apples = num_apples

        # +1 for surviving, +2 for eating an apple.
        self.score = 0
        self.snake = Snake()

        self.apple_locations = []
        for i in range(num_apples):
            self.apple_locations.append(self.new_apple_location())

    def win (self):
        return self.snake.length >= self.width * self.height

    def lose (self):
        # The game is lost if either:
        # The snake's head is outside the screen.
        return ((self.snake.head[0] < 0 or self.snake.head[0] >= self.width or
                 self.snake.head[1] < 0 or self.snake.head[1] >= self.height) or
        # Or if the snake's head is in its body.
                (self.snake.head in self.snake.body))
    
    def new_apple_location (self):
        apple_location = [random.randint(0, self.width - 1),
                          random.randint(0, self.height - 1)]
        max = 1000
        while (apple_location in self.apple_locations or
               apple_location in [self.snake.head] + self.snake.body):
            if (max <= 0):
                # TODO: Handle this better.
                print("No apple location found")
                exit()
            
            apple_location = [random.randint(0, self.width - 1),
                              random.randint(0, self.height - 1)]
            max -= 1
            
        return apple_location

    # Win loss detection is done from outside this method.
    def update (self, turn):
        self.score += 1
        # Turn the snake.
        self.snake.turn(turn)
        # Move the snake.
        apple_eaten = self.snake.move(self.apple_locations)

        # Create new apple if needed.
        if (apple_eaten):
            self.score += 2
            # If there is no space for a new apple, don't create a new one.
            if (self.num_apples + self.snake.length > self.width * self.height):
                self.num_apples -= 1
                self.apple_locations.remove(self.snake.head)
            else:
                # The head of the snake will be at the location of the eaten apple.
                self.apple_locations[self.apple_locations.index(self.snake.head)] = self.new_apple_location()
    
def main ():
    pygame.init()
    
    width = 6
    height = 6
    pixel_size = 20
    surface = pygame.display.set_mode((width * pixel_size, height * pixel_size))
    
    game = Game(width, height, 15)
    
    clock = pygame.time.Clock()
    fps = 5
            
    running = True
    
    while (running):
        turn = None
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False
                break;
            # Check if the user wants to turn the snake.
            elif (event.type == pygame.KEYDOWN):
                if (event.key == pygame.K_w or event.key == pygame.K_UP):
                    turn = directions[0]
                elif (event.key == pygame.K_a or event.key == pygame.K_LEFT):
                    turn = directions[3]
                elif (event.key == pygame.K_s or event.key == pygame.K_DOWN):
                    turn = directions[2]
                elif (event.key == pygame.K_d or event.key == pygame.K_RIGHT):
                    turn = directions[1]
        
        game.update(turn)
        if (game.win()):
            print("Win")
            game.score *= 100
            running = False
        if (game.lose()):
            game.score *= 0.7
            print("Lose")
            running = False

        surface.fill("black")
        # Draw the snake
        for s in [game.snake.head] + game.snake.body:
            rect = pygame.Rect((s[0] * pixel_size, s[1] * pixel_size), (pixel_size, pixel_size))
            pygame.draw.rect(surface, (0, 255, 0), rect)
        # Draw the apples.
        for a in game.apple_locations:
            rect = pygame.Rect((a[0] * pixel_size, a[1] * pixel_size), (pixel_size, pixel_size))
            pygame.draw.rect(surface, (255, 0, 0), rect)
            
        pygame.display.flip()
        
        delta = clock.tick(fps)

    print(f"The score was: {game.score}")
    pygame.quit()
            
if __name__ == "__main__":
    main()
