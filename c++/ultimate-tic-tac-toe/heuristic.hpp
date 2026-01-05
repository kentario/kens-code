#pragma once

#include <span>
#include <algorithm>
#include <bit>
#include <concepts>

#include "constants.hpp"
#include "game.hpp"

// Counts the number of winning moves by the first player on some tic-tac-toe board.
// Could be a macroboard, or could be a subboard.
size_t count_winning_moves (const uint16_t a, const uint16_t b) {
  size_t count {0};

  // For each empty square, count the number of filled two in a rows that correspond to it.
  for (const uint8_t square_i : Board::empty_squares[a | b]) {
    // For each 2 in a row corresponding with the empty square,
    for (const uint16_t two_in_a_row : TWO_IN_A_ROWS[square_i]) {
      // Check if it is filled by the player that is being counted.
      if ((two_in_a_row & a) == two_in_a_row) count++;
    }
  }

  return count;
}

// All heuristics should check for wins and draws.
#define HEURISTIC_BEGINNING						\
  /* Checkin for a win */						\
  for (const auto mask : WIN_MASKS) {					\
    if ((mask & board.macroboards[to_index(Role::MIN)]) == mask) return -100'000; \
    if ((mask & board.macroboards[to_index(Role::MAX)]) == mask) return  100'000; \
  }									\
  /* Checking for a draw. Draw if if all boards have been completed. */	\
  if ((board.macroboards[to_index(Player::X)] | board.macroboards[to_index(Player::O)] | board.macroboards[2]) == FULL_BOARD) return 0;

// Constants that are used in heuristics.
// Initialized with values chosen by me that seemed ok.
// These will eventually be changed to values optimized through evolution.
struct Eval_Params {
  // The value of capturing a specific type of subboard.
  double center {6};
  double corner {5};
  double edge {4};
  // The value of having more moves.
  // Would be mutiplied by the counted/estimated number of possible moves.
  double move_weight {0.2};
  // The importance of having multiple winning moves.
  // Will probably be used for a whole macroboard, where a winning move would just be winning a certain subboard to win the whole game.
  double win_options_weight {3};
  // How much should the state of an incomplete board matter?
  // High numbers should prioritize dominating specific boards, while low nubmers should mean that the state of specific incomplete subboards doesn't matter as much.
  double incomplete_subboard_weight {1};
};

// Only counts subboards that have been won.
double subboard_values_simple (const Board &board, const Eval_Params &params) {
  return
    (board.macroboards[to_index(Role::MAX)] & MOVE_MASKS[0] ? params.corner : 0) +
    (board.macroboards[to_index(Role::MAX)] & MOVE_MASKS[1] ? params.edge : 0) +
    (board.macroboards[to_index(Role::MAX)] & MOVE_MASKS[2] ? params.corner : 0) +
    (board.macroboards[to_index(Role::MAX)] & MOVE_MASKS[3] ? params.edge : 0) +
    (board.macroboards[to_index(Role::MAX)] & MOVE_MASKS[4] ? params.center : 0) +
    (board.macroboards[to_index(Role::MAX)] & MOVE_MASKS[5] ? params.edge : 0) +
    (board.macroboards[to_index(Role::MAX)] & MOVE_MASKS[6] ? params.corner : 0) +
    (board.macroboards[to_index(Role::MAX)] & MOVE_MASKS[7] ? params.edge : 0) +
    (board.macroboards[to_index(Role::MAX)] & MOVE_MASKS[8] ? params.corner : 0) -
    
    (board.macroboards[to_index(Role::MIN)] & MOVE_MASKS[0] ? params.corner : 0) -
    (board.macroboards[to_index(Role::MIN)] & MOVE_MASKS[1] ? params.edge : 0) -
    (board.macroboards[to_index(Role::MIN)] & MOVE_MASKS[2] ? params.corner : 0) -
    (board.macroboards[to_index(Role::MIN)] & MOVE_MASKS[3] ? params.edge : 0) -
    (board.macroboards[to_index(Role::MIN)] & MOVE_MASKS[4] ? params.center : 0) -
    (board.macroboards[to_index(Role::MIN)] & MOVE_MASKS[5] ? params.edge : 0) -
    (board.macroboards[to_index(Role::MIN)] & MOVE_MASKS[6] ? params.corner : 0) -
    (board.macroboards[to_index(Role::MIN)] & MOVE_MASKS[7] ? params.edge : 0) -
    (board.macroboards[to_index(Role::MIN)] & MOVE_MASKS[8] ? params.corner : 0);
}

// Returns 1 if max has won, -1 if min has won, and 0 otherwise.
// Only call if you know someone has won.
int check_winner (const Board &board) {
  for (const auto mask : WIN_MASKS) {
    if ((mask & board.macroboards[to_index(Role::MIN)]) == mask) return -1;
    if ((mask & board.macroboards[to_index(Role::MAX)]) == mask) return 1;
  }

  return 0;
}

double heur1 (const Board &board, [[maybe_unused]] const Eval_Params &params) {
  HEURISTIC_BEGINNING;
  // If max wins a board, +1, if min wins a board, -1.
  return std::popcount(board.macroboards[to_index(Role::MAX)]) - std::popcount(board.macroboards[to_index(Role::MIN)]);
}

double heur2 (const Board &board, const Eval_Params &params) {
  HEURISTIC_BEGINNING;
  
  return subboard_values_simple(board, params);
}

double heur3 (const Board &board, const Eval_Params &params) {
  HEURISTIC_BEGINNING;

  // More legal moves = better for the player about to play.
  // getting subboards is maybe more valuable then having more legal moves.
  return subboard_values_simple(board, params) + params.move_weight * sign(board.next_player()) * board.count_legal_moves();
}

// Maybe faster version of heur3
double heur4 (const Board &board, const Eval_Params &params) {
  HEURISTIC_BEGINNING;

  // If the move can be any board, that is good.
  return subboard_values_simple(board, params) + params.move_weight * sign(board.next_player()) * (board.forced_sb == ANY_SUBBOARD ? 50 : 4);
}

// Used for alpha beta pruning.
void sort_moves (const Board &board, std::span<Move> moves) {
  (void)board;
  (void)moves;
}

template <typename F>
concept Heuristic =
  std::invocable<F, const Board&, const Eval_Params&> &&
  std::convertible_to<
    std::invoke_result_t<F, const Board&, const Eval_Params&>,
    double>;
