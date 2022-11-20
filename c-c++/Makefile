build: sdl-hello-world temporary game-of-life

game-of-life: game-of-life.cpp
	g++ $< -o $@ -lSDL2

sdl-hello-world: sdl-hello-world.cpp
	g++ $< -o $@ -lSDL2

temporary: temporary.cpp
	g++ $< -o $@ -lSDL2

all: ende in-out

ende: ende.c
	gcc ende.c -o ende

in-out: in-out.c
	gcc in-out.c -o in-out

