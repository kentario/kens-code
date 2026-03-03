#include "simulate.hpp"

#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <tuple>
#include <exception>
#include <functional>
#include <algorithm>

#include "constants.hpp"
#include "game.hpp"
#include "bots.hpp"

std::ostream& operator<< (std::ostream &os, const Game_Record &game) {
  os << game.p0_name << " vs " << game.p1_name << '\n';
  os << "seed: " << game.seed << '\n';
  switch (game.result) {
  case Game_Result::PLAYER0_WIN:
    os << "p0 won";
    break;
  case Game_Result::PLAYER1_WIN:
    os << "p1 won";
    break;
  case Game_Result::DRAW:
    os << "draw";
    break;
  }

  return os;
}

Game_Record play_game (const Bot_ptr &p0, const Bot_ptr &p1, const uint64_t seed) {
  Game_Record game {p0->get_name(), p1->get_name(), seed};

  p0->reset(seed);
  p1->reset(seed);

  Board board {};
  Move move {};

  while (!board.terminal()) {
    //    std::cout << board << "\n\n";
    //    auto start = std::chrono::steady_clock::now();
    if (board.next_player() == Player::X) move = p0->pick_move(board);
    else move = p1->pick_move(board);
    //    auto end = std::chrono::steady_clock::now();
    //    if (std::chrono::duration<double, std::milli>(end - start).count() > 500) {
    //      save_positions("long position", &board, 1, true);
    //      std::cout << "saved long taking position\n";
    //    }
    //    std::cout << board.count_total_empty_squares() << " empty squares\n";
    //    std::cout << "chosen move " << move << '\n';

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

  // Since the game is over, the default value if no win is found should be a draw.
  game.result = Game_Result::DRAW;
  for (const auto mask : WIN_MASKS) {
    if ((mask & board.macroboards[to_index(Role::MIN)]) == mask) game.result = Game_Result::PLAYER0_WIN;
    if ((mask & board.macroboards[to_index(Role::MAX)]) == mask) game.result = Game_Result::PLAYER1_WIN;
  }

  return game;
}

Match_Stats& Match_Stats::operator+= (const Game_Record &game) {
  if (p0_name.empty()) {
    p0_name = game.p0_name;
    p1_name = game.p1_name;
  }
  num_games++;
  switch (game.result) {
  case Game_Result::PLAYER0_WIN:
    p0_wins++;
    break;
  case Game_Result::PLAYER1_WIN:
    p1_wins++;
    break;
  case Game_Result::DRAW:
    draws++;
    break;
  }

  return *this;
}

double Match_Stats::p0_win_rate () const { return static_cast<double>(p0_wins)/num_games; }
double Match_Stats::p1_win_rate () const { return static_cast<double>(p1_wins)/num_games; }
double Match_Stats::draw_rate () const { return static_cast<double>(draws)/num_games; }

std::ostream& operator<< (std::ostream& os, const Match_Stats &match) {
  os << "num games: " << match.num_games << '\n';
  os << match.p0_name << " win rate: " << match.p0_win_rate() << " (" << match.p0_wins << ")\n";
  os << match.p1_name << " win rate: " << match.p1_win_rate() << " (" << match.p1_wins << ")\n";
  os << "draw rate: " << match.draw_rate() << " (" << match.draws << ')';

  return os;
}

Tournament& Tournament::operator+= (const Game_Record &game) {
  // If the match entry doesn't exist, then a default one will be automatically constructed.
  // Doesn't matter that it's empty because the game will be pushed onto it immediately after.
  stats[{game.p0_name, game.p1_name}] += game;
  games.push_back(game);

  return *this;
}

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

  for (const auto &a : bots) {
    for (const auto &b : bots) {
      std::cout << "starting games for " << a->get_name() << " vs " << b->get_name() << '\n';
      for (size_t i {0}; i < games_per_pair; i++) {
	//	std::cout << i << ' ' << std::flush;
	tournament += play_game(a, b, seed_rng());
      }// std::cout << '\n';
    }
  }

  return tournament;
}
