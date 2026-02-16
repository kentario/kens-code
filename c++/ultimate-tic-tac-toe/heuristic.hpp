#pragma once

#include <concepts>
#include <functional>
#include <type_traits>

#include "game.hpp"

// Constants that are used in heuristics.
// Initialized with values chosen by me that seemed ok.
// These will eventually be changed to values optimized self-play.
struct Eval_Params {
  // The value of capturing a specific type of subboard.
  double center {6};
  double corner {5};
  double edge {4};
  // The value of having more moves.
  // Would be mutiplied by the counted/estimated number of possible moves.
  double move_weight {0.05};
  // The importance of having multiple winning moves.
  // Will probably be used for a whole macroboard, where a winning move would just be winning a certain subboard to win the whole game.
  double win_options_weight {3};
  // How much should the state of an incomplete board matter?
  // High numbers should prioritize dominating specific boards, while low nubmers should mean that the state of specific incomplete subboards doesn't matter as much.
  double incomplete_subboard_weight {1};
};

template <typename F>
concept Heuristic =
  std::invocable<F, const Board&, const Eval_Params&> &&
  std::convertible_to<
    std::invoke_result_t<F, const Board&, const Eval_Params&>,
    double>;

/*
  Counts the number of winning moves by the first player on some tic-tac-toe board.
  Could be a macroboard, or could be a subboard.
*/
size_t count_winning_moves (const uint16_t a, const uint16_t b);

// Only counts subboards that have been won.
double subboard_values_simple (const Board &board, const Eval_Params &params);

// What is at the beginning of most heuristics, check for win loss draw.
inline std::optional<double> terminal_check (const Board &board) {
  /* Checkin for a win */
  for (const auto mask : WIN_MASKS) {
    if ((mask & board.macroboards[to_index(Role::MIN)]) == mask) return LOSS;
    if ((mask & board.macroboards[to_index(Role::MAX)]) == mask) return WIN;
  }
  /* Checking for a draw. Draw if if all boards have been completed. */
  if ((board.macroboards[to_index(Player::X)] | board.macroboards[to_index(Player::O)] | board.macroboards[2]) == FULL_BOARD) return DRAW;

  // Not win or loss or draw => not terminal
  return {};
}

/*
  Returns 1 if max has won, -1 if min has won, and 0 otherwise.
  Only call if you know someone has won.
  Similar to terminal_score
*/
int check_winner (const Board &board, const size_t ply);

double heur1 (const Board &board, [[maybe_unused]] const Eval_Params &params);
double heur2 (const Board &board, const Eval_Params &params);
double heur3 (const Board &board, const Eval_Params &params);
// Maybe faster version of heur3
double heur4 (const Board &board, const Eval_Params &params);
