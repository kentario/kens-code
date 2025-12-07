#include <iostream>
#include <array>
#include <random>

#include "game.hpp"
#include "bots.hpp"

int main () {
  // Sample game against a randomly playing oponent.
  const std::array<Move, 45> moves {{
    {0, 0}, {0, 3}, {3, 2}, {2, 7}, {7, 4}, {4, 4}, {4, 1}, {1, 1}, {1, 0}, {0, 1}, {1, 3}, {3, 1}, {1, 6}, {6, 6}, {6, 1}, {6, 5}, {5, 1}, {3, 5}, {5, 8}, {8, 0}, {0, 4}, {4, 7}, {7, 2}, {2, 8}, {8, 8}, {8, 2}, {2, 6}, {6, 8}, {8, 1}, {4, 0}, {0, 8}, {8, 5}, {5, 7}, {7, 5}, {5, 6}, {6, 7}, {7, 7}, {7, 3}, {3, 4}, {4, 8}, {8, 3}, {3, 6}, {2, 2}, {2, 0}, {2, 4}
    }};
  
  Board board {};
  Random random {std::random_device{}()};
  Minimax minimax2 {2, heur1};
  Minimax minimax3 {3, heur1};
  Minimax minimax4 {4, heur1};
  Minimax_random mr2 {2, heur1};
  Minimax_random mr3 {3, heur1};
  Minimax_random mr4 {4, heur1};


  std::cout << board << "\n\n";

  Move move {};
  while (!terminal(board)) {
    std::cout << "moves played " << board.moves_played << '\n';

    if (board.next_player) {
      move = mr2(board);
      std::cout << "mr2 (maximizing) playing " << move << '\n';
    } else {
      move = mr3(board);
      std::cout << "mr3 (minimizing) playing " << move << '\n';
    }
    
    if (!play_move(board, move)) {
      std::cout << "Something has gone horribly wrong with trying to play the move " << move << '\n';
      break;
    }

    std::cout << board << "\n\n\n";
  }

  return 0;
}
