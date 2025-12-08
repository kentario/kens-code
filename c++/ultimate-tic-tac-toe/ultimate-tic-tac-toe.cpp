#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <memory>

#include "game.hpp"
#include "bots.hpp"

struct Game_info {
  std::vector<Move> moves;
  bool winner;
};

struct Match_Info {
  
};

template <typename T>
concept Bot_T = requires {
  std::is_base_of_v<Bot, T>;
};

using Bot_Pointer = std::unique_ptr<Bot>;

template <Bot_T A, Bot_T B>
// Each round consists of a game where A goes first, then B goes first, using the same random seed/starting position.
// The number of games played is 2 * num_rounds.
Match_Info match (const size_t num_rounds) {
  Match_Info res {};

  for (int i {0}; i < num_rounds; i++) {
    
  }
  
  return res;
}

int main () {
  std::mt19937 rng {};
  
  Board board {};
  Random random {rng};
  Minimax minimax1 {1, heur1};
  Minimax minimax4 {4, heur1};
  Minimax minimax5 {5, heur1};
  Minimax_random mr4 {4, heur1};
  Minimax_random mr6 {6, heur1, rng};
  Minimax_random mr9 {9, heur1, rng};


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

  std::cout << heur1(board) << '\n';

  return 0;
}
