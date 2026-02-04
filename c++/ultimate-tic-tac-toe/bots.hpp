#pragma once

#include <chrono>

#include <random>
#include <functional>
#include <array>
#include <vector>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <cfloat>

#include "constants.hpp"
#include "game.hpp"
#include "heuristic.hpp"

constexpr bool DEBUG {false};

struct Search_Stats {
  size_t nodes {};
  size_t cutoffs {};
};

class Bot {
private:
  const std::string name {};
  
public:
  Search_Stats stats {};
  
  Bot (const std::string &name) :
    name {name} {}

  // Returns the move that it wants to play
  virtual Move pick_move (const Board &board) = 0;

  // Would clear anything cached and other stuff, and also sets the seed.
  virtual void reset (const uint64_t) {}
  
  std::string get_name () const { return name; }
};

template <typename T>
concept Bot_T = requires {
  std::is_base_of_v<Bot, T>;
};

using Bot_ptr = std::unique_ptr<Bot>;

class Random : public Bot {
protected:
  std::mt19937 rng {};
  
public:
  Random (const std::string &name, const uint64_t seed = 0) :
    Bot {name}, rng {seed} {}
  
  void reset (const uint64_t seed = 0) override { rng.seed(seed); }
  
  virtual Move pick_move (const Board &board) override {
    std::vector<Move> moves {board.legal_moves()};
    if (board.terminal()) return {9, 9};
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
  }
};

template <Heuristic H>
class Minimax : public Bot {
protected:
  const size_t max_depth;
  Eval_Params params;
  const H eval;
  std::mt19937 rng;

public:
  Minimax (const std::string &name, const size_t max_depth,
	   const Eval_Params &params, const H &eval,
	   const uint64_t seed = 0) :
    Bot {name}, max_depth {max_depth},
    params {params}, eval {eval},
    rng {seed} {}
  
  void reset (const uint64_t seed = 0) override {
    rng.seed(seed);
    stats = Search_Stats {};
  }
  
  double minimax (const Board &board, size_t depth, double alpha, double beta) {
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

  Move pick_move (const Board &board) override {
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
};

template <Heuristic H>
class Negamax : public Bot {
protected:
  const size_t max_depth;
  Eval_Params params;
  const H eval;
  std::mt19937 rng;

public:
  Negamax (const std::string &name, const size_t max_depth,
	   const Eval_Params &params, const H &eval,
	   const uint64_t seed = 0) :
    Bot {name}, max_depth {max_depth},
    params {params}, eval {eval},
    rng {seed} {}
  
  void reset (const uint64_t seed = 0) override {
    rng.seed(seed);
    stats = Search_Stats {};
  }

  // Always evaluating from the perspecive of the current person about to play.
  double negamax (const Board &board, size_t depth, double alpha, double beta) {
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

  Move pick_move (const Board &board) override {
    if (board.terminal()) return {9, 9};

    std::vector<Move> moves {board.legal_moves()};
    std::shuffle(moves.begin(), moves.end(), rng);
    // Index and value of the best move.
    // If the current player is X, they are trying to minimize, so start with the biggest possible value.
    // Vice versa for O.
    std::pair<size_t, double> best_move {-1, -DBL_MAX};
    for (size_t i {0}; i < moves.size(); i++) {
      double value {-negamax(board.play_move_unsafe_value(moves[i]), max_depth - 1, -DBL_MAX, DBL_MAX)};
      std::cout << moves[i] << ": " << value << '\n';

      if (value > best_move.second) best_move = {i, value};
    }
    
    return moves[best_move.first];
  }

  double full_search (const Board &board, const size_t ply, double alpha, double beta) {
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
  
  Move pick_move_full (const Board &board) {
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
};

template <Heuristic H, Heuristic S>
class Negamax_Ordered : public Bot {
protected:
  const size_t max_depth;
  Eval_Params params;
  const H eval;
  const S sort;
  std::mt19937 rng;

public:
  Negamax_Ordered (const std::string &name, const size_t max_depth,
		   const Eval_Params &params, const H &eval, const S &sort,
		   const uint64_t seed = 0) :
    Bot {name}, max_depth {max_depth},
    params {params}, eval {eval}, sort {sort},
    rng {seed} {}
  
  void reset (const uint64_t seed = 0) override {
    rng.seed(seed);
    stats = Search_Stats {};
  }

  // Always evaluating from the perspecive of the current person about to play.
  double negamax (const Board &board, size_t depth, double alpha, double beta) {
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

  Move pick_move (const Board &board) override {
    if (board.terminal()) return {9, 9};

    std::vector<Move> moves {board.legal_moves()};
    std::shuffle(moves.begin(), moves.end(), rng);
    // Index and value of the best move.
    // If the current player is X, they are trying to minimize, so start with the biggest possible value.
    // Vice versa for O.
    std::pair<size_t, double> best_move {-1, -DBL_MAX};
    for (size_t i {0}; i < moves.size(); i++) {
      double value {-negamax(board.play_move_unsafe_value(moves[i]), max_depth - 1, -DBL_MAX, DBL_MAX)};
      std::cout << moves[i] << ": " << value << '\n';

      if (value > best_move.second) best_move = {i, value};
    }
    
    return moves[best_move.first];
  }

  void sort_moves (std::span<Move> moves, const Board &board) {
    std::array<std::array<size_t, 9>, 9> values {};
    for (const auto move : moves) {
      values[move.subboard][move.square] = sort(board.play_move_unsafe_value(move), params);
    }

    std::sort(moves.begin(), moves.end(), [&values](const Move a, const Move b) {
      return values[a.subboard][a.square] > values[b.subboard][b.square];
    });
  }
  
  double full_search (const Board &board, const size_t ply, double alpha, double beta) {
    // ply is the distance to the root node.
    stats.nodes++;
    if (board.terminal()) return check_winner(board, ply) * sign(board.next_player());

    std::vector<Move> moves {board.legal_moves()};

    if (ply <= 2) {
      sort_moves(moves, board);
    }

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
  
  Move pick_move_full (const Board &board) {
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
};
