#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include <tuple>
#include <exception>
#include <functional>
#include <algorithm>

#include "game.hpp"
#include "bots.hpp"

enum class GAME_RESULT {
  PLAYER0_WIN,
  PLAYER1_WIN,
  DRAW
};

struct Game_Record {
  std::string p0_name {};
  std::string p1_name {};
  
  uint64_t seed {};
  std::vector<Move> moves {};
  GAME_RESULT result {};
};

std::ostream& operator<< (std::ostream &os, const Game_Record &game) {
  os << game.p0_name << " vs " << game.p1_name << '\n';
  os << "seed: " << game.seed << '\n';
  switch (game.result) {
  case GAME_RESULT::PLAYER0_WIN:
    os << "p0 won";
    break;
  case GAME_RESULT::PLAYER1_WIN:
    os << "p1 won";
    break;
  case GAME_RESULT::DRAW:
    os << "draw";
    break;
  }

  return os;
}

Game_Record play_game (const Bot_ptr& p0, const Bot_ptr& p1, const uint64_t seed) {
  Game_Record game {p0->get_name(), p1->get_name(), seed};

  p0->reset(seed);
  p1->reset(seed);
	       
  Board board {};
  Move move {};

  while (!board.terminal()) {
    if (board.next_player() == Player::X) move = (*p0)(board);
    else move = (*p1)(board);
    
    if (!board.play_move(move)) {
      const std::string msg {
	to_string(move) +
	" played by " +
	(board.next_player() == Player::X ?
	 (p0->get_name() + " against " + p1->get_name()) :
	 (p1->get_name() + " against " + p0->get_name())) +
	" on board\n" +
	to_string(board) +
	"\n"
      };
      
      throw std::domain_error {msg};
    }

    game.moves.push_back(move);
  }

  switch (heur1(board)) {
  case -10000:
    game.result = GAME_RESULT::PLAYER0_WIN;
    break;
  case 10000:
    game.result = GAME_RESULT::PLAYER1_WIN;
    break;
  default:
    // Since the game is over (terminal(board) == false), then it must be a draw.
    game.result = GAME_RESULT::DRAW;
  };
  
  return game;
}

// The player/bot going first is always the same in a match.
// p0 goes first and p1 goes second.
struct Match_Stats {
  std::string p0_name {};
  std::string p1_name {};

  size_t num_games {0};
  size_t p0_wins {0};
  size_t p1_wins {0};
  size_t draws {0};

  Match_Stats& operator+= (const Game_Record &game) {
    if (p0_name.empty()) {
      p0_name = game.p0_name;
      p1_name = game.p1_name;
    }
    num_games++;
    switch (game.result) {
    case GAME_RESULT::PLAYER0_WIN:
      p0_wins++;
      break;
    case GAME_RESULT::PLAYER1_WIN:
      p1_wins++;
      break;
    case GAME_RESULT::DRAW:
      draws++;
      break;
    }

    return *this;
  }

  double p0_win_rate () const { return static_cast<double>(p0_wins)/num_games; }
  double p1_win_rate () const { return static_cast<double>(p1_wins)/num_games; }
  double draw_rate () const { return static_cast<double>(draws)/num_games; }
};

std::ostream& operator<< (std::ostream& os, const Match_Stats &match) {
  os << "num games: " << match.num_games << '\n';
  os << match.p0_name << " win rate: " << match.p0_win_rate() << " (" << match.p0_wins << ")\n";
  os << match.p1_name << " win rate: " << match.p1_win_rate() << " (" << match.p1_wins << ")\n";
  os << "draw rate: " << match.draw_rate() << " (" << match.draws << ')';

  return os;
}

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
  
  Tournament& operator+= (const Game_Record &game) {
    // If the match entry doesn't exist, then a default one will be automatically constructed.
    // Doesn't matter that it's empty because the game will be pushed onto it immediately after.
    stats[{game.p0_name, game.p1_name}] += game;
    games.push_back(game);

    return *this;
  }
};

std::ostream& operator<< (std::ostream &os, const Tournament t) {
  for (const auto &[names, stat] : t.stats) {
    os << names.first << " vs " << names.second << ":\n";
    os << stat << "\n\n";
  }

  return os;
}

Tournament simulate (const std::span<const Bot_ptr> bots, const size_t games_per_pair) {
  std::mt19937 seed_rng {std::random_device{}()};
  Tournament tournament {};

  for (const Bot_ptr &a : bots) {
    for (const Bot_ptr &b : bots) {
      for (size_t i {0}; i < games_per_pair; i++) {
	tournament += play_game(a, b, seed_rng());
      }
    }
  }

  return tournament;
}
