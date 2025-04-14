cmake_minimum_required(VERSION 3.22)

project(learn-cmake)

set(CMAKE_CXX_STANDARD 23)

# If the code was in a src/ directory, then use src/main.cpp and src/adder.cpp
add_executable(learn-cmake
	main.cpp
	adder.cpp
)

target_link_libraries(learn-cmake sfml-graphics sfml-window sfml-system)
