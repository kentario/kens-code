#pragma once

#include <iostream>
#include <chrono>
#include <string>
#include <random>
#include <span>
#include <algorithm>
#include <array>

#include "game.hpp"
#include "heuristic.hpp"
#include "bots.hpp"

// For checking minimaxfull efficiency/validity.
template <Bot_T B>
// The bot must be able to do a full search.
requires requires (B bot, const Board board) { bot.pick_move_full(board); }
void test_full_search (std::mt19937 &rng, B bot) {
  std::array<Board, 1000> boards_sorted {};
  for (int i {0}; i < 1'000; i++) {
    boards_sorted[i] = random_position(rng, false, 10, 81);
  }
  std::sort(boards_sorted.begin(), boards_sorted.end(), [](const Board &a, const Board &b) {
    return a.count_total_empty_squares() > b.count_total_empty_squares();
  });

  for (int i {999}; i >= 0; i--) {
    std::cout << i << '\n';
    std::cout << player(boards_sorted[i].next_player()) << " (" << role(boards_sorted[i].next_player()) << ") to play\n";
    std::cout << boards_sorted[i] << "\n";
    std::cout << "there are " << boards_sorted[i].count_total_empty_squares() << " empty squares in the board\n";
    auto m = bot.pick_move_full(boards_sorted[i]);
    std::cout << m << "\n\n\n";
    if (!boards_sorted[i].is_legal(m)) {
      std::cerr << "super bad stuff";
      std::cerr << i << " " << m;
      throw std::domain_error {
	"illegal move by " + bot.get_name() + " on board\n" + to_string(boards_sorted[i])
      };
    }
  }
}
