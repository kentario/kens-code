#include <iostream>
#include <random>
#include <exception>
#include <memory>

#include "game.hpp"
#include "bots.hpp"
#include "simulate.hpp"
#include "benchmark.hpp"

int main () {
  try {
    std::mt19937 rng {};
    // std::array<Board, 100'000> boards {};
    // for (Board &b : boards) {
    //   b = random_position(rng, false);
    // }
    // save_positions("positions-100k", boards.data(), boards.size(), false);
    
    std::vector<Board> boards {load_positions("positions-100k", 1000)};
    Bot_ptr a = std::make_unique<Minimax>("m4h1", 6, heur1);
    Bot_ptr b = std::make_unique<Minimax>("m4h2", 6, heur2);
    Bot_ptr c = std::make_unique<Minimax>("m4h3", 6, heur3);
    // std::cout << benchmark_bot_move_generation(*a, boards) << '\n';
    // std::cout << benchmark_bot_move_generation(*b, boards) << '\n';
    // std::cout << benchmark_bot_move_generation(*c, boards) << '\n';

    size_t num_games {200};
    std::vector<Bot_ptr> bots;
    bots.push_back(std::move(a));
    bots.push_back(std::move(b));
    bots.push_back(std::move(c));
    std::cout << simulate(bots, num_games) << '\n';
    
  } catch (const std::exception &e) {
    std::cerr << e.what();
    
    return EXIT_FAILURE;
  }
  
  return 0;
}
