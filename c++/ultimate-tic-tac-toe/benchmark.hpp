#pragma once

#include <iostream>
#include <string>
#include <string_view>
#include <chrono>
#include <random>

#include "game.hpp"
#include "bots.hpp"

Board random_position (std::mt19937 rng, const bool terminal_allowed, const size_t min_moves = 20, const size_t max_moves = 40) {
  Board board {};
  Random r {"", rng()};

  if (min_moves > max_moves) {
    throw std::domain_error {"Minimum number of moves is greater than the maximum"};
  }
  
  std::uniform_int_distribution<size_t> dist {min_moves, max_moves};
  const size_t target_num_moves {dist(rng)};

  bool board_found {false};
  while (!board_found) {
    board = Board {};

    size_t moves_played {0};
    for (; moves_played < target_num_moves; moves_played++) {
      if (terminal(board)) break;

      const Move move {r(board)};
      
      if (!play_move(board, move)) {
	const std::string msg {
	  to_string(move) +
	  " played by random on board\n" +
	  to_string(board) +
	  "\n"
	};
      
	throw std::domain_error {msg};
      }
    }

    // If terminal boards are allowed, then the board just has to have reached close enough to the target number of moves.
    // If terminal boards are not allowed, then the board also can't be terminal.
    if (terminal_allowed) {
      board_found = (moves_played >= min_moves);
    } else {
      board_found = (moves_played >= min_moves && !terminal(board));
    }
  }

  return board;
}

struct Benchmark_Result {
  std::string name;
  size_t positions;
  double ms;
  double ms_per_position;
};

std::ostream& operator<< (std::ostream &os, const Benchmark_Result &res) {
  os << res.name << " played " << res.positions << " positions,\n";
  os << "taking " << res.ms << " milliseconds, or " << res.ms_per_position << " ms per position";
  return os;
};

Benchmark_Result benchmark_bot_move_generation (Bot &bot, std::span<Board> positions) {
  auto start = std::chrono::steady_clock::now();

  for (const auto &p : positions) {
    volatile Move m __attribute__((unused)) {bot(p)};
  }
  
  auto end = std::chrono::steady_clock::now();
  double ms {std::chrono::duration<double, std::milli>(end - start).count()};

  return Benchmark_Result {
    .name = bot.get_name(),
    .positions = positions.size(),
    .ms = ms,
    .ms_per_position = ms/positions.size()
  };
}

Benchmark_Result benchmark_heuristic (Heuristic h, const std::string &name, std::span<Board> positions) {
  auto start = std::chrono::steady_clock::now();

  for (const auto &p : positions) {
    volatile int m __attribute__((unused)) {h(p)};
  }
  
  auto end = std::chrono::steady_clock::now();
  double ms {std::chrono::duration<double, std::milli>(end - start).count()};

  return Benchmark_Result {
    .name = name,
    .positions = positions.size(),
    .ms = ms,
    .ms_per_position = ms/positions.size()
  };
}


// TODO benchmark passing board by value vs reference.
// TODO benchmark legal_moves taking a function to apply to each legal move as it is found, instead of returning a vector. This should make it so that the moves aren't iterated over twice.
