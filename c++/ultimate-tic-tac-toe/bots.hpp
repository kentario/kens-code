#pragma once

#include <span>
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

struct Search_Stats {
  size_t nodes {};
  size_t cutoffs {};
};

class Bot {
private:
  const std::string name {};
  
public:
  Search_Stats stats {};
  
  Bot (const std::string &name);

  virtual ~Bot () = default;

  // Returns the move that it wants to play
  virtual Move pick_move (const Board &board) = 0;

  // Would clear anything cached and other stuff, and also sets the seed.
  virtual void reset (const uint64_t seed = 0);
  
  std::string get_name () const;
};

template <typename T>
concept Bot_T = std::is_base_of_v<Bot, T>;

using Bot_ptr = std::unique_ptr<Bot>;

class Random : public Bot {
protected:
  std::mt19937 rng {};
  
public:
  Random (const std::string &name, const uint64_t seed = 0);
  
  void reset (const uint64_t seed = 0) override;
  virtual Move pick_move (const Board &board) override;
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
	   const uint64_t seed = 0);

  void reset (const uint64_t seed = 0) override;
  void set_params (const Eval_Params &new_params);
  
  double minimax (const Board &board, size_t depth, double alpha, double beta);
  Move pick_move (const Board &board) override;
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
	   const uint64_t seed = 0);
  
  void reset (const uint64_t seed = 0) override;

  void set_params (const Eval_Params &new_params);

  // Always evaluating from the perspecive of the current person about to play.
  double negamax (const Board &board, size_t depth, double alpha, double beta);

  Move pick_move (const Board &board) override;

  double full_search (const Board &board, const size_t ply, double alpha, double beta);
  
  Move pick_move_full (const Board &board);
};

#include "bots.tpp"
