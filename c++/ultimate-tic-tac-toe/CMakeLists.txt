cmake_minimum_required(VERSION 3.22)

project(ultimate-tic-tac-toe)

set(CMAKE_CXX_STANDARD 23)

add_executable(ultimate-tic-tac-toe
	ultimate-tic-tac-toe.cpp
	constants.cpp
	game.cpp
	heuristic.cpp
	bots.cpp
	benchmark.cpp
	simulate.cpp
	tune.cpp
)
