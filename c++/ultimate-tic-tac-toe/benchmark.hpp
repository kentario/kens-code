#pragma once

#include <iostream>
#include <string>
#include <chrono>
#include <random>

#include "game.hpp"
#include "bots.hpp"

Board random_position (std::mt19937 rng, const size_t min_moves = 20, const size_t max_moves = 40, const bool terminal_allowed) {
  Board board {};
  Random r {"", seed};

  std::uniform_int_distribution<size_t> dist {min_depth, max_depth};
  const size_t target_num_moves {dist(rng)};

  bool board_found {false};
  while (!board_found) {
    board = Board {};
    for (size_t i {0}; i < target_num_moves; i++) {
      if (terminal(board)) break;

      play_move(board, r(board));
    }

    // If terminal boards are allowed, then the board just has to have reached the target number of moves.
    // If terminal boards are not allowed, then the board also can't be terminal.
    if (terminal_allowed) {
      board_found = i >= min_depth;
    } else {
      board_found = i >= min_depth && !terminal(board);
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
  os << name << " played " << positions << " positions,\n";
  os << "taking " << ms << " milliseconds, or " << ms_per_position << " ms per position";
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
    .ms_per_position = ms/positions.size()};
}

Benchmark_Result benchmark_heuristic (Heuristic h, std::span<Board> positions) {
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
    .ms_per_position = ms/positions.size()};
}
