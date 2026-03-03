#pragma once

#include <iostream>
#include <chrono>
#include <vector>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <tuple>
#include <exception>
#include <functional>
#include <algorithm>

#include "constants.hpp"
#include "game.hpp"
#include "bots.hpp"

struct Game_Record {
  std::string p0_name {};
  std::string p1_name {};
  
  uint64_t seed {};
  std::vector<Move> moves {};
  Game_Result result {};
};

std::ostream& operator<< (std::ostream &os, const Game_Record &game);

Game_Record play_game (const Bot_ptr &p0, const Bot_ptr &p1, const uint64_t seed);

// The player/bot going first is always the same in a match.
// p0 goes first and p1 goes second.
struct Match_Stats {
  std::string p0_name {};
  std::string p1_name {};

  size_t num_games {0};
  size_t p0_wins {0};
  size_t p1_wins {0};
  size_t draws {0};

  Match_Stats& operator+= (const Game_Record &game);

  double p0_win_rate () const;
  double p1_win_rate () const;
  double draw_rate () const;
};

std::ostream& operator<< (std::ostream& os, const Match_Stats &match);

// For Hashing with a pair of strings.
// https://stackoverflow.com/questions/32685540/why-cant-i-compile-an-unordered-map-with-a-pair-as-key
// Posted by Antony Hatchkins, modified by community. See post 'Timeline' for change history
// Retrieved 2025-12-11, License - CC BY-SA 4.0

// from boost (functional/hash):
// see http://www.boost.org/doc/libs/1_35_0/doc/html/hash/combine.html
template <typename T>
inline void hash_combine (size_t &seed, T const &v) {
  seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct Pair_Hash {
  template <typename T1, typename T2>
  size_t operator() (const std::pair<T1, T2> &p) const {
    size_t seed {0};
    hash_combine(seed, p.first);
    hash_combine(seed, p.second);
    
    return seed;
  }
};

struct Tournament {
  // TODO: remove vector games, make match stats have a pointer to each game played in the match.
  std::vector<Game_Record> games {};

  // (p0, p1) => Match_Stats
  std::unordered_map<
    std::pair<std::string, std::string>,
    Match_Stats,
    Pair_Hash> stats;
  
  Tournament& operator+= (const Game_Record &game);
};

std::ostream& operator<< (std::ostream &os, const Tournament t);

Tournament simulate (const std::span<const Bot_ptr> bots, const size_t games_per_pair);
