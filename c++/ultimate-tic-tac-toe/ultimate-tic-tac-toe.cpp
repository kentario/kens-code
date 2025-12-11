#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <utility>

#include "game.hpp"
#include "bots.hpp"

enum class Game_Result {
  PLAYER0_WIN,
  PLAYER1_WIN,
  DRAW
};

struct Game_Record {
  std::string p0_name {};
  std::string p1_name {};
  
  uint64_t seed {};
  std::vector<Move> moves {};
  Game_Result result {};
};

// The player/bot going first is always the same in a match.
// p0 goes first and p1 goes second.
struct Match_Stats {
  std::string p0_name {};
  std::string p1_name {};

  size_t num_games {0};
  size_t p0_wins {0};
  size_t p1_wins {0};
  size_t draws {0};

  Match_Stats& add_game (const Game_Record &game) {
    num_games++;
    switch (result) {
    case PLAYER0_WIN:
      p0_wins++;
      break;
    case PLAYER1_WIN:
      p1_wins++;
      break;
    case DRAW:
      draws++;
      break;
    }

    return *this;
  }

  Match_Stats& operator+= (const Game_Record &game) { return add_game(game); }
  
  double p0_win_rate () const { return static_cast<double>(p0_wins)/num_games; }
  double p1_win_rate () const { return static_cast<double>(p1_wins)/num_games; }
  double draw_rate () const { return static_cast<double>(draws)/num_games; }
};

struct Tournament {
  std::vector<Game_Record> games {};

  // (p0, p1) => Match_Stats
  std::unordered_map<std::pair<std::string, std::string>, Match_Stats> stats {};

  Tournament& add_game (const Game_Record &game) {
    // If the match entry doesn't exist, then a default one will be automatically constructed.
    stats[{game.p0_name, p1_name}] += game;
    games.push_back(game);
  }

  Tournament& operator+= (const Game_Record &game) { return add_game(game); }
};

int main () {
  std::mt19937 rng {};
  
  //  Board board {};
  Random random {rng};
  Minimax minimax1 {1, heur1};
  Minimax minimax4 {4, heur1};
  Minimax minimax5 {5, heur1};
  Minimax_Random mr4 {4, heur1};
  Minimax_Random mr6 {6, heur1, rng};
  Minimax_Random mr9 {9, heur1, rng};

  std::cout << "Same bots\n";
  std::cout << match(std::make_unique<Minimax_Random>(5, heur1, rng),
		     std::make_unique<Minimax_Random>(5, heur1, rng),
		     1000);
  std::cout << "Same b using heur2\n";
  std::cout << match(std::make_unique<Minimax_Random>(5, heur1, rng),
		     std::make_unique<Minimax_Random>(5, heur2, rng),
		     1000);
  
  /*
  std::cout << board << "\n\n";
  Move move {};
  while (!terminal(board)) {
    std::cout << "moves played " << board.moves_played << '\n';

    if (board.next_player) {
      move = mr9(board);
      std::cout << "mr9 (maximizing) playing " << move << '\n';
    } else {
      move = minimax5(board);
      std::cout << "minimax5 (minimizing) playing " << move << '\n';
    }

    // if (board.moves_played == 42) {
    //   move = {7, 8};
    //   std::cout << "overriding bot, playing " << move << '\n';
    // }
    
    if (!play_move(board, move)) {
      std::cout << "Something has gone horribly wrong with trying to play the move " << move << '\n';
      break;
    }

    std::cout << board << "\n\n\n";
  }
  */

  return 0;
}
