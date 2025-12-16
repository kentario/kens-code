#include <iostream>
#include <random>
#include <exception>
#include <memory>

#include "game.hpp"
#include "bots.hpp"
#include "simulate.hpp"

static std::random_device rng {};

int main () {
  try {
    Bot_Pointer a {std::make_unique<Minimax_Random>("mr5h1", 5, heur1, rng())};
    Bot_Pointer b {std::make_unique<Minimax_Random>("mr5h1", 5, heur2, rng())};

    Match_Stats a_first {};
    Match_Stats b_first {};
    std::cout << "Starting...\n";
    for (int i {0}; i < 1000; i++) {
      if (i == 1000/2 - 1) std::cout << "Halfway\n";
      a_first += play_game(a.get(), b.get(), rng());
      b_first += play_game(b.get(), a.get(), rng());
    }
    
    std::cout << "\na first:\n";
    std::cout << a_first << "\n\n";
    std::cout << "b first:\n";
    std::cout << b_first << '\n';
    

    /*
    Minimax_Random mr4 {4, heur1};
    Minimax_Random mr6 {6, heur1, rng};
    Minimax_Random mr9 {9, heur1, rng};

    std::cout << "Same bots\n";
    std::cout << match(std::make_unique<Minimax_Random>(5, heur1, rng),
		       std::make_unique<Minimax_Random>(5, heur1, rng),
		       1000);

    std::cout << "Same b using heur2\n";
    std::cout << match(std::make_unique<Minimax_Random>(5, heur1, rng),
		       std::make_unique<Minimax_Random>(5, heur2, rng),
		       1000);
    */
  } catch (const std::exception &e) {
    std::cerr << e.what();
    
    return EXIT_FAILURE;
  }
  
  return 0;
}
