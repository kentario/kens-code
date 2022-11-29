build: sdl-hello-world mandelbrot game-of-life minesweeper key-mouse-sdl2

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

