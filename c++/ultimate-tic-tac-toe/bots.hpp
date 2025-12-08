#pragma once

#include <bit>
#include <random>
#include <functional>
#include <array>
#include <vector>
#include <algorithm>

#include "game.hpp"

class Bot {
public:
  // Returns the move that it wants to play
  virtual Move operator() (const Board board) = 0;
};

class Random : public Bot {
protected:
  std::mt19937 rng;
public:
  // Template forwarding constructor.
  // Takes any arguments that mt19937 can take in its constructor, and forwards them to it.
  template<typename... Types>
  Random (Types&&... args)
    : rng {std::forward<Types>(args)...} {}
  
  Move operator() (const Board board) {
    std::vector<Move> moves {legal_moves(board)};
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
  }
};

// For later on if I get confused.
// https://www.learncpp.com/cpp-tutorial/function-pointers/
using Heuristic = std::function<int(Board)>;

class Minimax : public Bot {
protected:
  const size_t max_depth;
  Heuristic eval;
public:
  Minimax () = delete;
  
  Minimax (size_t max_depth, Heuristic eval) :
    max_depth {max_depth}, eval {eval} {}
  
  int minimax (Board board, size_t depth, int alpha, int beta) {
    if (terminal(board) || depth <= 0) return eval(board);

    // If it is the min player's turn, then they will try to minimize the evaluation of the board.
    // X is trying to minimize, O is trying to maximize.
    int value;
    if (board.next_player == MIN) {
      // X's turn
      // Trying to minimize, so start with the biggest possible value and find better and better eevaluations.
      value = std::numeric_limits<int>::max();
      for (const auto &move : legal_moves(board)) {
	value = std::min(value, minimax(play_move_unsafe_value(board, move), depth - 1, alpha, beta));
	beta = std::min(value, beta);
	if (beta <= alpha) break;
      }
    } else {
      // O's turn
      value = std::numeric_limits<int>::min();
      for (const auto &move : legal_moves(board)) {
	value = std::max(value, minimax(play_move_unsafe_value(board, move), depth - 1, alpha, beta));
	alpha = std::max(value, alpha);
	if (beta <= alpha) break;
      }
    }

    return value;
  }
  
  Move operator() (const Board board) {
    auto moves = legal_moves(board);
    // Index and value of the best move.
    // If the current player is X, they are trying to minimize, so start with the biggest possible value.
    // Vice versa for O.
    std::array<int, 2> best_move {-1, board.next_player == MAX ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max()};
    for (size_t i {0}; i < moves.size(); i++) {
      int value {minimax(play_move_unsafe_value(board, moves[i]), max_depth - 1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max())};

      std::cout << moves[i].subboard << moves[i].square << ": " << value << '\n';
      
      // If the current player wants to maximize, they want value to be higher than the best found so far.
      if ((board.next_player == MAX && value > best_move[1]) ||
	  (board.next_player == MIN && value < best_move[1]))
	best_move = {static_cast<int>(i), value};
    }

    return moves[best_move[0]];
  }
};

// Same as minimax, except instead of playing the first best move it sees, randomly picks from the best moves.
class Minimax_random : public Minimax {
protected:
  std::mt19937 rng;
public:
  
  template <typename... Types>
  Minimax_random (size_t max_depth, Heuristic eval, Types&&... args) :
    Minimax {max_depth, eval}, rng {std::forward<Types>(args)...} {}
  
  Move operator() (const Board board) {
    auto moves = legal_moves(board);
    // Shuffle the moves beforehand so that if there is a tie, the best one is picked randomly.
    std::shuffle(moves.begin(), moves.end(), rng);
    
    // Index and value of the best move.
    // If the current player is X, they are trying to minimize, so start with the biggest possible value.
    // Vice versa for O.
    std::array<int, 2> best_move {-1, board.next_player == MAX ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max()};
    for (size_t i {0}; i < moves.size(); i++) {
      int value {minimax(play_move_unsafe_value(board, moves[i]), max_depth - 1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max())};

      std::cout << moves[i].subboard << moves[i].square << ": " << value << '\n';
      
      // If the current player wants to maximize, they want value to be higher than the best found so far.
      if ((board.next_player == MAX && value > best_move[1]) ||
	  (board.next_player == MIN && value < best_move[1]))
	best_move = {static_cast<int>(i), value};
    }

    return moves[best_move[0]];
  }
};

int heur1 (const Board board) {
  // Check if there is a win
  for (const auto mask : WIN_MASKS) {
    // Player 0 (X) is minimizing.
    if ((mask & board.macroboards[X]) == mask) return -10000;
    if ((mask & board.macroboards[O]) == mask) return  10000;
  }

  // If max wins a board, +1, if min wins a board, -1.
  return std::popcount(board.macroboards[O]) - std::popcount(board.macroboards[X]);
}
