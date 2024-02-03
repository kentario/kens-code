build: sdl-hello-world mandelbrot game-of-life minesweeper key-mouse-sdl2 draw neural-network othello connect-4 3d-game tutorial count-unique-digits pathfinding test-osmium functions

functions: functions.cpp
	g++ -g $< -o $@

test-osmium: test-osmium.cpp
	g++ -g $< -o $@ -lz -lSDL2 -pthread -lxml2 -lexpat

run-pathfinding: pathfinding
	./pathfinding

pathfinding: pathfinding.cpp
	g++ -g $< -o $@ -std=c++20 -lSDL2

count-unique-digits: count-unique-digits.cpp
	g++ -g $< -o $@

tutorial: tutorial.cpp
	g++ -g $< -o $@

3d-game: 3d-game.cpp
	g++ -g $< -o $@ -lSDL2

connect-4: connect-4.cpp
	g++ -g $< -o $@

othello: othello.cpp
	g++ $< -o $@

run-neural-network: neural-network
	./neural-network

neural-network: neural-network.cpp
	g++ -g $< -o $@

draw: draw.cpp
	g++ $< -o $@ -lSDL2

key-mouse-sdl2: key-mouse-sdl2.cpp
	g++ $< -o $@ -lSDL2

minesweeper: minesweeper.cpp
	g++ $< -o $@ -lSDL2

game-of-life: game-of-life.cpp
	g++ $< -o $@ -lSDL2

sdl-hello-world: sdl-hello-world.cpp
	g++ $< -o $@ -lSDL2

mandelbrot: mandelbrot.cpp
	g++ $< -o $@ -lSDL2

all: ende in-out

ende: ende.c
	gcc ende.c -o ende

in-out: in-out.c
	gcc in-out.c -o in-out

