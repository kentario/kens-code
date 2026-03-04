#pragma once

#include <type_traits>
#include <random>
#include <array>
#include <span>
#include <vector>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <cfloat>

#include "constants.hpp"
#include "game.hpp"
#include "heuristic.hpp"

template <size_t max_depth, Heuristic eval>
Minimax<max_depth, eval>::Minimax (const std::string &name,
		     const Eval_Params params,
		     const uint64_t seed) :
  Bot {name, params}, rng {seed} {}

template <size_t max_depth, Heuristic eval>
void Minimax<max_depth, eval>::reset (const uint64_t seed) {
  rng.seed(seed);
  stats = Search_Stats {};
}

template <size_t max_depth, Heuristic eval>
double Minimax<max_depth, eval>::minimax (const Board &board, size_t depth, double alpha, double beta) {
  stats.nodes++;
  if (board.terminal() || depth <= 0) return eval(board, params);

  // If it is the min player's turn, then they will try to minimize the evaluation of the board.
  // X is trying to minimize, O is trying to maximize.
  double value;
  std::vector<Move> moves {board.legal_moves()};

  if (is_min(board.next_player())) {
    // X's turn
    // Trying to minimize, so start with the biggest possible value and find better and better evaluations.
    value = DBL_MAX;
    for (const auto &move : moves) {
      value = std::min(value, minimax(board.play_move_unsafe_value(move), depth - 1, alpha, beta));
      beta = std::min(value, beta);
      if (beta <= alpha) {
	stats.cutoffs++;
	break;
      }
    }
  } else {
    // O's turn
    value = -DBL_MAX;
    for (const auto &move : moves) {
      value = std::max(value, minimax(board.play_move_unsafe_value(move), depth - 1, alpha, beta));
      alpha = std::max(value, alpha);
      if (beta <= alpha) {
	stats.cutoffs++;
	break;
      }
    }
  }

  return value;
}

template <size_t max_depth, Heuristic eval>
Move Minimax<max_depth, eval>::pick_move (const Board &board) {
  if (board.terminal()) return {9, 9};
  std::vector<Move> moves {board.legal_moves()};

  // Index and value of the best move.
  // If the current player is X, they are trying to minimize, so start with the biggest possible value.
  // Vice versa for O.
  std::pair<size_t, double> best_move {-1, is_max(board.next_player()) ? -DBL_MAX : DBL_MAX};
  for (size_t i {0}; i < moves.size(); i++) {
    double value {minimax(board.play_move_unsafe_value(moves[i]), max_depth - 1, -DBL_MAX, DBL_MAX)};
    //      std::cout << "(" << moves[i] << ", " << value << ") ";

    // If the current player wants to maximize, they want value to be higher than the best found so far.
    if ((is_max(board.next_player()) && value > best_move.second) ||
	(is_min(board.next_player()) && value < best_move.second))
      best_move = {i, value};
  }

  return moves[best_move.first];
}

template <size_t max_depth, Heuristic eval>
Negamax<max_depth, eval>::Negamax (const std::string &name,
		     const Eval_Params params,
		     const uint64_t seed) :
  Bot {name, params}, rng {seed} {}

template <size_t max_depth, Heuristic eval>
void Negamax<max_depth, eval>::reset (const uint64_t seed) {
  rng.seed(seed);
  stats = Search_Stats {};
}

template <size_t max_depth, Heuristic eval>
double Negamax<max_depth, eval>::negamax (const Board &board, size_t depth, double alpha, double beta) {
  stats.nodes++;
  // Always returns from the perspective of the current player.
  // Multiplies by -1 if the player is min, as then small (good) numbers become big.
  if (board.terminal() || depth <= 0) return eval(board, params) * sign(board.next_player());

  std::vector<Move> moves {board.legal_moves()};
  // Maybe order the moves here.
  double value {-DBL_MAX};
  for (const Move m : moves) {
    // -negamax because what is high (good) for the other player should be bad (low) for the current player.
    value = std::max(value, -negamax(board.play_move_unsafe_value(m), depth - 1, -beta, -alpha));
    alpha = std::max(alpha, value);
    if (alpha >= beta) {
      stats.cutoffs++;
      break;
    }
  }

  return value;
}

template <size_t max_depth, Heuristic eval>
Move Negamax<max_depth, eval>::pick_move (const Board &board) {
  if (board.terminal()) return {9, 9};

  std::vector<Move> moves {board.legal_moves()};
  std::shuffle(moves.begin(), moves.end(), rng);
  // Index and value of the best move.
  // If the current player is X, they are trying to minimize, so start with the biggest possible value.
  // Vice versa for O.
  std::pair<size_t, double> best_move {-1, -DBL_MAX};
  for (size_t i {0}; i < moves.size(); i++) {
    double value {-negamax(board.play_move_unsafe_value(moves[i]), max_depth - 1, -DBL_MAX, DBL_MAX)};

    //      std::cout << moves[i] << ": " << value << '\n';

    if (value > best_move.second) best_move = {i, value};
  }

  return moves[best_move.first];
}

template <size_t max_depth, Heuristic eval>
double Negamax<max_depth, eval>::full_search (const Board &board, const size_t ply, double alpha, double beta) {
  // ply is the distance to the root node.
  stats.nodes++;
  if (board.terminal()) return check_winner(board, ply) * sign(board.next_player());

  std::vector<Move> moves {board.legal_moves()};

  double value {-DBL_MAX};
  for (const Move m : moves) {
    value = std::max(value, -full_search(board.play_move_unsafe_value(m), ply + 1, -beta, -alpha));
    alpha = std::max(alpha, value);
    if (alpha >= beta) {
      stats.cutoffs++;
      break;
    }
  }

  return value;
}

template <size_t max_depth, Heuristic eval>
Move Negamax<max_depth, eval>::pick_move_full (const Board &board) {
  if (board.terminal()) return {9, 9};

  std::vector<Move> moves {board.legal_moves()};
  std::pair<size_t, double> best_move {-1, -DBL_MAX};
  // Currently at root node.
  constexpr size_t ply {0};
  for (size_t i {0}; i < moves.size(); i++) {
    double value {-full_search(board.play_move_unsafe_value(moves[i]), ply + 1, -DBL_MAX, DBL_MAX)};
    std::cout << moves[i] << ": " << value << '\n';

    if (value > best_move.second) best_move = {i, value};
  }

  return moves[best_move.first];
}
