build: custom-operators compile-time-prime hello-world pathfinding

custom-operators: custom-operators.cpp
	g++ $< -o $@

compile-time-prime: compile-time-prime.cpp
	g++ -ftemplate-depth=2000 $< -o $@

hello-world: hello-world.cpp
	g++ $< -o $@

run-pathfinding: pathfinding
	./pathfinding

pathfinding: pathfinding.cpp
	g++ -g $< -o $@ -std=c++20 -lSDL2

