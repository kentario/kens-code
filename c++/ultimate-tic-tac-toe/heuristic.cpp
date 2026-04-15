#include "heuristic.hpp"

#include <iostream>
#include <span>
#include <optional>
#include <algorithm>
#include <bit>

#include "constants.hpp"
#include "game.hpp"

std::ostream& operator<< (std::ostream &os, const Eval_Params params) {
  os << "{" << std::endl;
  os << "    center: " << params.center << std::endl;
  os << "    corner: " << params.corner << std::endl;
  os << "    edge: " << params.edge << std::endl;
  os << "    move_weight: " << params.move_weight << std::endl;
  os << "    win_options_weight: " << params.win_options_weight << std::endl;
  os << "    incomplete_subboard_weight: " << params.incomplete_subboard_weight << std::endl;
  os << "}";
  
  return os;
}

// Counts the number of winning moves by the first player on some tic-tac-toe board.
// Could be a macroboard, or could be a subboard.
size_t count_winning_moves (const uint16_t a, const uint16_t b) {
  size_t count {0};

  // For each empty square, count the number of filled two in a rows that correspond to it.
  for (const uint8_t square_i : Board::empty_squares[a | b]) {
    // For each 2 in a row corresponding with the empty square,
    for (size_t j {0}; j < TWO_IN_A_ROW_COUNTS[square_i]; j++) {
      const uint16_t two_in_a_row {TWO_IN_A_ROWS_ARR[square_i][j]};
      // Check if it is filled by the player that is being counted.
      if ((two_in_a_row & a) == two_in_a_row) count++;
    }
  }

  return count;
}

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
int check_winner (const Board &board, const size_t ply) {
  for (const auto mask : WIN_MASKS) {
    if ((mask & board.macroboards[to_index(Role::MIN)]) == mask) return LOSS + ply;
    if ((mask & board.macroboards[to_index(Role::MAX)]) == mask) return WIN - ply;
  }

  return 0;
}

double heur1 (const Board &board, [[maybe_unused]] const Eval_Params &params) {
  // If the game is done, then return the value of the game.
  if (auto t = terminal_check(board)) return *t;
  
  // If max wins a board, +1, if min wins a board, -1.
  return std::popcount(board.macroboards[to_index(Role::MAX)]) - std::popcount(board.macroboards[to_index(Role::MIN)]);
}

double heur2 (const Board &board, const Eval_Params &params) {
  if (auto t = terminal_check(board)) return *t;

  return subboard_values_simple(board, params);
}

double heur3 (const Board &board, const Eval_Params &params) {
  if (auto t = terminal_check(board)) return *t;

  // More legal moves = better for the player about to play.
  // getting subboards is maybe more valuable then having more legal moves.
  return subboard_values_simple(board, params) + params.move_weight * sign(board.next_player()) * board.count_legal_moves();
}

// Maybe faster version of heur3
// not much faster.
double heur4 (const Board &board, const Eval_Params &params) {
  if (auto t = terminal_check(board)) return *t;

  // If the move can be any board, that is good.
  return subboard_values_simple(board, params) + params.move_weight * sign(board.next_player()) * (board.forced_sb == ANY_SUBBOARD ? 50 : 4);
}

double heur5 (const Board &board, const Eval_Params &params) {
  if (auto t = terminal_check(board)) return *t;

  return subboard_values_simple(board, params) +
    params.move_weight * sign(board.next_player()) * board.count_legal_moves() -
    count_winning_moves(board.macroboards[to_index(Role::MIN)], static_cast<uint16_t>(board.macroboards[to_index(Role::MAX)] | board.macroboards[2])) +
    count_winning_moves(board.macroboards[to_index(Role::MAX)], static_cast<uint16_t>(board.macroboards[to_index(Role::MIN)] | board.macroboards[2]));
}
