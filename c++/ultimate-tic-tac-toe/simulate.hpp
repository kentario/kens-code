#pragma once

#include <iostream>
#include <vector>
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

template <Bot_T A, Bot_T B>
Game_Record play_game (A &p0, B &p1, const uint64_t seed) {
  Game_Record game {p0.get_name(), p1.get_name(), seed};
  // If they are the same name, then label them a and b.
  if (game.p0_name == game.p1_name) {
    game.p0_name += "_a";
    game.p1_name += "_b";
  }

  p0.set_seed(seed);
  p1.set_seed(seed);
	       
  Board board {};
  Move move {};

  while (!terminal(board)) {
    if (board.next_player == X) move = p0(board);
    else move = p1(board);
    
    if (!play_move(board, move)) {
      const std::string msg {
	"Illegal move " +
	to_string(move) +
	" played by " +
	(board.next_player == X ?
	 (p0.get_name() + " against " + p1.get_name()) :
	 (p1.get_name() + " against " + p0.get_name())) +
	" on board\n" +
	to_string(board) +
	"\n"
      };
      
      throw std::domain_error {msg};
    }

    game.moves.push_back(move);
  }

  switch (Heur1::eval(board)) {
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

template <Bot_T... Bs>
Tournament simulate (const size_t games_per_round) {
  std::mt19937 seed_rng {std::random_device{}()};
  
  Tournament tournament {};

  std::tuple<Bs...> bots {};

  for (size_t i {0}; i < games_per_round; i++) {
    std::apply([&](auto&... a) {
      // For each bot,
      (std::apply([&a, &tournament, &seed_rng](auto&... b) {
	// Play it against every other bot.
	((tournament += play_game(a, b, seed_rng())), ...);
      }, bots), ...);
    }, bots);
  }

  return tournament;
}
