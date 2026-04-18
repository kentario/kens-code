#pragma once

#include <span>
#include <type_traits>
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

protected:
  Eval_Params params {};

public:
  Search_Stats stats {};

  Bot () = default;
  
  Bot (const std::string &name, const Eval_Params params = Eval_Params {});

  virtual ~Bot () = default;

  // Returns the move that it wants to play
  // Would clear anything cached and other stuff, and also sets the seed.
  virtual void reset (const uint64_t seed = 0);
  
  virtual Move pick_move (const Board &board) = 0;

  std::string get_name () const;
  void set_params (const Eval_Params &new_params);
  Eval_Params get_params () const;
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

template <size_t max_depth, Heuristic eval>
class Minimax : public Bot {
protected:
  std::mt19937 rng;

public:
  Minimax (const std::string &name,
	   const Eval_Params params = Eval_Params {},
	   const uint64_t seed = 0);

  void reset (const uint64_t seed = 0) override;
  
  double minimax (const Board &board, size_t depth, double alpha, double beta);
  Move pick_move (const Board &board) override;
};

template <size_t max_depth, Heuristic eval>
class Negamax : public Bot {
protected:
  std::mt19937 rng;

public:
  Negamax (const std::string &name,
	   const Eval_Params params = Eval_Params {},
	   const uint64_t seed = 0);
  
  void reset (const uint64_t seed = 0) override;

  // Always evaluating from the perspecive of the current person about to play.
  double negamax (Board &board, size_t depth, double alpha, double beta);

  Move pick_move (const Board &board) override;

  double full_search (Board &board, const size_t ply, double alpha, double beta);
  
  Move pick_move_full (const Board &board);
};

#include "bots.tpp"
