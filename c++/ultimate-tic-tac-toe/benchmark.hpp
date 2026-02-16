#pragma once

#include <iostream>
#include <chrono>
#include <random>
#include <string>
#include <span>
#include <algorithm>
#include <array>

#include "game.hpp"
#include "heuristic.hpp"
#include "bots.hpp"

Board random_position (std::mt19937 &rng, const bool terminal_allowed, const size_t min_moves = 20, const size_t max_moves = 40);

struct Benchmark_Result {
  std::string name;
  size_t positions;
  size_t nodes;
  size_t cutoffs;
  double ms;
  double ms_per_position;
  double ms_per_node;
};

std::ostream& operator<< (std::ostream &os, const Benchmark_Result &res);

Benchmark_Result benchmark_bot_move_generation (Bot &bot, std::span<Board> positions);

Benchmark_Result benchmark_bot_nodes (Bot &bot, std::span<Board> positions);

template <Heuristic H>
Benchmark_Result benchmark_heuristic (H h, const std::string &name, std::span<Board> positions);

Benchmark_Result benchmark_find_legal_moves (std::span<Board> positions);

// For checking minimaxfull efficiency/validity.
template <Bot_T B>
// The bot must be able to do a full search.
requires requires (B bot, const Board board) { bot.pick_move_full(board); }
void test_full_search (std::mt19937 &rng, B bot);

#include "benchmark.tpp"
