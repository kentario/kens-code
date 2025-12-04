#include <iostream>
#include <vector>
#include <array>
#include <string>

const int X {1};
const int O {-1};

// Check if someone has won by doing board_winner(state)
class Board_ {
private:
  // Left to right, top to bottom.
  // 1 for X, -1 for O.
  std::array<std::array<int, 9>, 9> board {};

  // The state of each board. 1 means X won, -1 means O won, and 0 means no one has one.
  std::array<int, 9> state {};

  // The location on a mini board of the last move played.
  // 0 means next move must be on board[0], 1 on board[1], etc.
  // 9 means it's the beginning of the game.
  std::array<size_t, 2> last_move {9, 9};

  // even => X to play, odd => O to play.
  int moves_played {};

  // Checks if any board filled with -1, 0, 1 has a 3 in a row, and returns the type that has the 3 in a row.
  // no one has won/draw => 0
  // X has won => 1, O has won => -1
  int board_winner (const std::array<int, 9> &b) const {
    // Horizontal
    for (int i {0}; i < 3; i++) {
      // If all three in a row are equal and not 0, then there is a win.
      // Same logic for other checks.
      if (b[3 * i] == b[3 * i + 1] && b[3 * i] == b[3 * i + 2] && b[3 * i] != 0) return b[3 * i];
    }
    // Vertical
    for (int i {0}; i < 3; i++) {
      if (b[i] == b[i + 3] && b[i] == b[i + 6] && b[i] != 0) return b[i];
    }

    // Diagonal
    if (b[0] == b[4] && b[0] == b[8] && b[0] != 0) return b[0];
    if (b[2] == b[4] && b[2] == b[6] && b[0] != 0) return b[2];


    // Fix bug where board winner is 0 on a stalemate board, but also 0 on an uncompleted board.
    
    return 0;
  }

public:
  bool is_legal (const size_t board_number, const size_t square) const {
    //    std::cout << "((" << (board_number == last_move[1]) << " || " << (state[last_move[1]] != 0) << " || " << (moves_played == 0) << ") && ";
    //    std::cout << (board[board_number][square] == 0) << " && " << (board_number < 9 && square < 9) << ")\n";
    
    //       playing on the correct board or it's been won            o    r it's a stalemate    or it's the beginning
    return ((board_number == last_move[1] || state[last_move[1]] != 0 || stalemate(last_move[1]) || moves_played == 0)
	    // and   it's an empty square       and it's within the board
	    && board[board_number][square] == 0 && board_number < 9 && square < 9);
  }

  // I thought it would be nicer to split these up into separate methods so that when calling it from outside it doesn't require passing in an array that it already owns.
  int subboard_winner (const size_t b) const {
    return board_winner(board[b]);
  }

  bool stalemate (const size_t b) const {
    // If the board state isn't 0, then someone has one.
    if (state[b] != 0) return false;
    // If the board state is 0, then it has to be full, otherwise it's not done.
    for (const auto e : board[b]) {
      if (e == 0) return false;
    }

    return true;
  }

  int game_winner () const {
    return board_winner(state);
  }
  
  // Returns true if the move succeeeded, false if it failed.
  bool play_move (const size_t board_number, const size_t square) {
    if (!is_legal(board_number, square)) return false;

    /*
      moves_played = even  moves_played = odd
      0  1
      0  -2
      1 -1
     */
    //    std::cout << "move " << board_number << " " << square << ": " << board[board_number][square] << '\n';
    board[board_number][square] = -2 * (moves_played++ & 1) + 1;
    //    std::cout << "move " << board_number << " " << square << ": " << board[board_number][square] << '\n';
    last_move = {board_number, square};

    state[board_number] = subboard_winner(board_number);

    return true;
  }

  bool play_move (const std::array<int, 2> &move) {
    return play_move(move[0], move[1]);
  }

  std::array<int, 9> operator[] (const size_t i) const {return board[i];}
  std::array<std::array<int, 9>, 9> get_board () const {return board;}

  std::array<int, 9> get_state () const {return state;}

  std::array<size_t, 2> get_last_move () const {return last_move;}
};

std::string print_board_raw (const Board_ &board) {
  std::string res {};

  for (int row {0}; row < 9; row++) {
    for (int col {0}; col < 9; col++) {
      const int v {board[(col/3) + (row/3) * 3][(col % 3) + (row % 3) * 3]};
      res += std::to_string(v);
      
      if (col != 8) {
	if (col % 3 == 2) {
	  res += "  ||  ";
	} else {
	  res += '|';
	}
      }
    }
    if (row != 8) {
      if (row % 3 == 2) {
	res += "\n       ||         ||";
	res += "\n-------++---------++-------";
	res += "\n-------++---------++-------";
	res += "\n       ||         ||\n";
      } else {
	res += "\n-+-+-  ++  -+-+-  ++  -+-+-\n";
      }
    }
  }
  
  return res;
}


std::string print_board (const Board_ &board) {
  std::string res {};

  for (int row {0}; row < 9; row++) {
    for (int col {0}; col < 9; col++) {
      const int v {board[(col/3) + (row/3) * 3][(col % 3) + (row % 3) * 3]};
      res += v > 0 ? 'X' : (v < 0 ? 'O' : ' ');
      
      if (col != 8) {
	if (col % 3 == 2) {
	  res += "  ||  ";
	} else {
	  res += '|';
	}
      }
    }
    if (row != 8) {
      if (row % 3 == 2) {
	res += "\n       ||         ||";
	res += "\n-------++---------++-------";
	res += "\n-------++---------++-------";
	res += "\n       ||         ||\n";
      } else {
	res += "\n-+-+-  ++  -+-+-  ++  -+-+-\n";
      }
    }
  }

  // For each winning board, make a big version of the shape on top.
  const std::string big_x {"\\   /                        \\ /                          X                          / \\                        /   \\"};
  const std::string big_o {" /^\\                        |   |                       |   |                       |   |                        \\_/ "};
  size_t top_left = 0;
  for (size_t i {0}; i < 9; i++) {
    const int s {board.get_state()[i]};
    
    switch (s) {
    case 0:
      break;
    case 1:
      for (size_t j {0}; j < big_x.size(); j++) {
	if (j % 28 < 5) {
	  res[top_left + j] = big_x[j];
	}
      }
      break;
    case -1:
      for (size_t j {0}; j < big_o.size(); j++) {
	if (j % 28 < 5) {
	  res[top_left + j] = big_o[j];
	}
      }
    }

    top_left += 11;
    if (i % 3 == 2) {
      top_left += 205;
    }
  }

  return res;
}

constexpr uint16_t WIN_MASKS [8] {
  // Horizontal
  0b111000000,
  0b000111000,
  0b000000111,
  // Vertical
  0b100100100,
  0b010010010,
  0b001001001,
  // Diagonal
  0b100010001,
  0b001001001
};

struct Board {
  // For both, [0] => X, [1] => O, and for macroboards, [2] => draw
  // 9 boards, left to right top to bottom, and 1 for each player.
  uint16_t subboards[9][2] {};
  // 1 overall board for each player, stores where a player has won a subboard, and the last board is the ones that have ended in a draw.
  uint16_t macroboards[3] {};

  // -1 means any subboard is allowed.
  int forced_sb {-1};
};

void make_move (Board board, const size_t subboard, const size_t square) {
}

int main () {
  Board_ board {};

  // Sample game against a randomly playing oponent.
  const std::vector<std::array<int, 2>> moves {
    {0, 0}, {0, 3}, {3, 2}, {2, 7}, {7, 4}, {4, 4}, {4, 1}, {1, 1}, {1, 0}, {0, 1}, {1, 3}, {3, 1}, {1, 6}, {6, 6}, {6, 1}, {6, 5}, {5, 1}, {3, 5}, {5, 8}, {8, 0}, {0, 4}, {4, 7}, {7, 2}, {2, 8}, {8, 8}, {8, 2}, {2, 6}, {6, 8}, {8, 1}, {4, 0}, {0, 8}, {8, 5}, {5, 7}, {7, 5}, {5, 6}, {6, 7}, {7, 7}, {7, 3}, {3, 4}, {4, 8}, {8, 3}, {3, 6}, {2, 2}, {2, 0}, {2, 4}
  };

  std::cout << print_board(board) << '\n';

  return 0;
}

