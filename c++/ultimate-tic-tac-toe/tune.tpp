#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <utility>
#include <string>
#include <algorithm>
#include <random>

#include "game.hpp"
#include "bots.hpp"
#include "heuristic.hpp"
#include "simulate.hpp"

template <Bot_T B>
Eval_Params evolve_eval_params_for_bot (const size_t generations, const size_t gen_size) {
  std::mt19937 rng {};

  Eval_Params current_best {};

  Bot_ptr benchmark_bot {std::make_unique<B>("benchmark")};
  
  std::vector<Bot_ptr> bots;
  for (size_t i {0}; i < gen_size; i++) {
    // Bots named by their index.
    bots.push_back(std::make_unique<B>(std::to_string(i)));
    bots[bots.size() - 1]->reset(rng());
  }

  for (size_t gen {0}; gen < generations; gen++) {
    // Make first bot have the best parameters, and the others all are slightly tweaked.
    bots[0]->set_params(current_best);
    for (size_t i {1}; i < gen_size; i++) {
      bots[i]->set_params(tweak_params(current_best));
    }

    std::cout << "current best before gen " << gen << '\n';
    std::cout << current_best << '\n';
    std::cout << "Benchmark against benchmark_bot:\n";
    Tournament temp {};
    for (size_t i {0}; i < 20; i++) {
      temp += play_game(benchmark_bot, bots[0], rng());
      temp += play_game(bots[0], benchmark_bot, rng());
    }
    std::cout << temp << '\n';

    // Play the generation
    auto res = simulate(bots, 4);

    // Find the bot with the most wins.
    size_t current_most_wins {0};
    // For each bot,
    for (size_t i {0}; i < gen_size; i++) {
      size_t wins {0};
      // Count the number of wins it had.
      for (size_t j {0}; j < gen_size; j++) {
	wins += res.stats[{std::to_string(i), std::to_string(j)}].p0_wins;
      }
      std::cout << "bot " << i << " had " << wins << " wins\n";

      if (wins > current_most_wins) {
	current_most_wins = wins;
	current_best = bots[i]->get_params();
      }
    }
  }

  return current_best;
}
