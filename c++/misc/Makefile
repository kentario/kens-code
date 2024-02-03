build: custom-operators compile-time-prime hello-world

custom-operators: custom-operators.cpp
	g++ $< -o $@

compile-time-prime: compile-time-prime.cpp
	g++ -ftemplate-depth=2000 $< -o $@

hello-world: hello-world.cpp
	g++ $< -o $@
