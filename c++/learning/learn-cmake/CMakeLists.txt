cmake_minimum_required(VERSION 3.22.1)

project(learn-cmake)

add_executable(learn-cmake main.cpp)

target_link_libraries(learn-cmake sfml-graphics sfml-window sfml-system)
