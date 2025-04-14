build: count-unique-digits 3d-game compile-time-prime connect-4 draw game-of-life hello-world key-mouse-sdl2 mandelbrot minesweeper neural-network othello pathfinding sdl-hello-world test-osmium tutorial eulers-method

3d-game: 3d-game.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20 -lSDL2

compile-time-prime: compile-time-prime.cpp
	g++ -ftemplate-depth=2000 $< -o $@ -Wall -Wextra

connect-4: connect-4.cpp
	g++ -g -O3 $< -o $@ -Wall -Wextra -std=c++20

count-unique-digits: count-unique-digits.cpp
	g++ -g -O3 $< -o $@ -Wall -Wextra -std=c++20

draw: draw.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20 -lSDL2

game-of-life: game-of-life.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20 -lSDL2

hello-world: hello-world.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20

key-mouse-sdl2: key-mouse-sdl2.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20 -lSDL2

mandelbrot: mandelbrot.cpp
	g++ -g -O3 $< -o $@ -Wall -Wextra -std=c++20 -lSDL2

minesweeper: minesweeper.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20 -lSDL2

neural-network: neural-network.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20

othello: othello.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20

pathfinding: pathfinding.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20 -lSDL2

sdl-hello-world: sdl-hello-world.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20 -lSDL2

# TODO: Find what needs to be linked for osmium.
test-osmium: test-osmium.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20

tutorial: tutorial.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20

eulers-method: eulers-method.cpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20
