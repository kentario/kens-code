run-main:main
	./main

main: main.cpp expression.hpp expression-factory.hpp token.hpp lexer.hpp parser.hpp
	g++ -g $< -o $@ -Wall -Wextra -std=c++20
