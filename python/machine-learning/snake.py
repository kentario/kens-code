import random
import numpy
import pygame

# up     right    down    left  straight
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
        self.direction = directions[1]

    # None can be used to represent no turn.
    def turn (self, direction):
        # If the components add to 0, then it is a 180 degree turn, which is the same as no turn.
        if (direction is None or
            self.direction[0] + direction[0] == 0 or
            self.direction[1] + direction[1] == 0):
            return
        
        self.direction = direction

    # Returns true if an apple was eaten, false if not.
    def move (self, apples):
        self.body = numpy.vstack([self.head, self.body])
        self.head += self.direction

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
        self.score = 0
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
    def update (self, turn):
        self.score += 1
        # Turn the snake.
        self.snake.turn(turn)
        # Move the snake.
        apple_eaten = self.snake.move(self.apple_locations)

        # Create new apple if needed.
        if (apple_eaten):
            self.score += 2
            self.apple_locations = remove(self.apple_locations, self.snake.head)
            # If there is no space for a new apple, don't create a new one.
            if (self.num_apples + self.snake.length > self.width * self.height):
                self.num_apples -= 1
            else:
                # If there is space for an apple, make one.
                self.apple_locations = numpy.vstack([self.apple_locations, self.new_apple_location()])

def main ():
    pygame.init()
    
    width = 6
    height = 6
    pixel_size = 20
    surface = pygame.display.set_mode((width * pixel_size, height * pixel_size))
    
    game = Game(width, height, 20)
    
    clock = pygame.time.Clock()
    fps = 3

    num_epochs = 1000
    
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
            game.score *= 100
            running = False
        if (game.lose()):
            game.score *= 0.7
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

    print(f"The score was: {game.score}")
    pygame.quit()
            
if __name__ == "__main__":
    main()
