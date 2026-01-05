#pragma once

#include <iostream>
#include <string>
#include <string_view>
#include <chrono>
#include <random>
#include <span>
#include <algorithm>

#include "game.hpp"
#include "heuristic.hpp"
#include "bots.hpp"

Board random_position (std::mt19937 &rng, const bool terminal_allowed, const size_t min_moves = 20, const size_t max_moves = 40) {
  Board board {};
  Random r {"", rng()};

  if (min_moves > max_moves) {
    throw std::domain_error {"Minimum number of moves is greater than the maximum"};
  }
  if (min_moves > 81 || max_moves > 81) {
    throw std::domain_error {"More than 81 moves cannot be played."};
  }
  
  std::uniform_int_distribution<size_t> dist {min_moves, max_moves};
  size_t target_num_moves {dist(rng)};

  bool board_found {false};
  size_t tries {0};
  while (!board_found) {
    tries++;
    board = Board {};

    if (tries >= 50) {
      target_num_moves = std::max(min_moves, target_num_moves - 1);
    }

    size_t moves_played {0};
    for (; moves_played < target_num_moves; moves_played++) {
      if (board.terminal()) break;

      const Move move {r(board)};
      
      if (!board.play_move(move)) {
	const std::string msg {
	  to_string(move) +
	  " played by random on board\n" +
	  to_string(board) +
	  "\n"
	};
      
	throw std::domain_error {msg};
      }
    }

    if (terminal_allowed) {
      board_found = moves_played >= min_moves;
    } else {
      board_found = moves_played >= min_moves && !board.terminal();
    }
  }

  return board;
}

struct Benchmark_Result {
  std::string name;
  size_t positions;
  size_t nodes;
  size_t cutoffs;
  double ms;
  double ms_per_position;
  double ms_per_node;
};

std::ostream& operator<< (std::ostream &os, const Benchmark_Result &res) {
  os << "{" << std::endl;
  os << "    " << res.name << " played " << res.positions << " positions in " << res.ms << " milliseconds" << std::endl;
  os << "    " << res.ms_per_position << " ms per position" << std::endl;
  os << "    searched " << res.nodes << " nodes and pruned " << res.cutoffs << " times" << std::endl;
  os << "    " << res.ms_per_node << " ms per node" << std::endl;
  os << "}";
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
    .nodes = bot.stats.nodes,
    .cutoffs = bot.stats.cutoffs,
    .ms = ms,
    .ms_per_position = ms/positions.size(),
    .ms_per_node = ms/bot.stats.nodes
  };
}

Benchmark_Result benchmark_bot_nodes (Bot &bot, std::span<Board> positions) {
  auto start = std::chrono::steady_clock::now();

  for (const auto &p : positions) {
    volatile Move m __attribute__((unused)) {bot(p)};
  }
  
  auto end = std::chrono::steady_clock::now();
  double ms {std::chrono::duration<double, std::milli>(end - start).count()};

  return Benchmark_Result {
    .name = bot.get_name(),
    .positions = positions.size(),
    .nodes = bot.stats.nodes,
    .cutoffs = bot.stats.cutoffs,
    .ms = ms,
    .ms_per_position = ms/positions.size(),
    .ms_per_node = ms/bot.stats.nodes
  };
}

template <Heuristic H>
Benchmark_Result benchmark_heuristic (H h, const std::string &name, std::span<Board> positions) {
  auto start = std::chrono::steady_clock::now();

  for (const auto &p : positions) {
    [[maybe_unused]] volatile double m {h(p, Eval_Params {})};
  }
  
  auto end = std::chrono::steady_clock::now();
  double ms {std::chrono::duration<double, std::milli>(end - start).count()};

  return Benchmark_Result {
    .name = name,
    .positions = positions.size(),
    .nodes = 0,
    .cutoffs = 0,
    .ms = ms,
    .ms_per_position = ms/positions.size(),
    .ms_per_node = 0
  };
}

Benchmark_Result benchmark_find_legal_moves (std::span<Board> positions) {
  auto start = std::chrono::steady_clock::now();

  for (const auto &p : positions) {
    volatile std::vector<Move> moves __attribute__((unused)) {p.legal_moves()};
  }
  
  auto end = std::chrono::steady_clock::now();
  double ms {std::chrono::duration<double, std::milli>(end - start).count()};

  return Benchmark_Result {
    .name = "legal_moves",
    .positions = positions.size(),
    .nodes = 0,
    .cutoffs = 0,
    .ms = ms,
    .ms_per_position = ms/positions.size(),
    .ms_per_node = 0
  };
}
