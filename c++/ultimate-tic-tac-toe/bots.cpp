#include "bots.hpp"

#include <random>
#include <vector>
#include <string>

#include "game.hpp"
#include "heuristic.hpp"

Bot::Bot (const std::string &name, const Eval_Params params) :
  name {name}, params {params} {}

// Would clear anything cached and other stuff, and also sets the seed.
// Should not reset eval params.
void Bot::reset (const uint64_t) {}
std::string Bot::get_name () const { return name; }
void Bot::set_params (const Eval_Params &new_params) { params = new_params; }
Eval_Params Bot::get_params () const { return params; }

Random::Random (const std::string &name, const uint64_t seed) :
  Bot {name}, rng {seed} {}
  
void Random::reset (const uint64_t seed) { rng.seed(seed); }
  
Move Random::pick_move (const Board &board) {
  std::vector<Move> moves {board.legal_moves()};
  if (board.terminal()) return {9, 9};
  std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
  return moves[dist(rng)];
}
