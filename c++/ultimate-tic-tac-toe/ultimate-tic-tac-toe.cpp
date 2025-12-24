#include <iostream>
#include <random>
#include <exception>
#include <memory>

#include "game.hpp"
#include "bots.hpp"
#include "simulate.hpp"

int main () {
  try {
    std::cout << simulate<Random, Minimax<4, Heur1>, Minimax<4, Heur2>>(100) << '\n';
    
  } catch (const std::exception &e) {
    std::cerr << e.what();
    
    return EXIT_FAILURE;
  }
  
  return 0;
}
