#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <random>
#include <functional>
#include <limits>
#include <bit>
#include <algorithm>

std::string to_string (std::array<std::array<int, 9>, 9> board, std::array<int, 9> state) {
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
    const int s {state[i]};
    
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
  0b001010100
};

constexpr uint16_t MOVE_MASKS [9] {
  0b100000000,
  0b010000000,
  0b001000000,
  0b000100000,
  0b000010000,
  0b000001000,
  0b000000100,
  0b000000010,
  0b000000001
};

constexpr uint16_t FULL_BOARD {
  0b111111111
};

struct Move {
  size_t subboard {};
  size_t square {};
};

struct Board {
  // For both, [0] => X, [1] => O, and for macroboards, [2] => draw
  // 9 boards, left to right top to bottom, and 1 for each player.
  // 0b100000000 is just the top left cell
  uint16_t subboards[9][2] {};
  // 1 overall board for each player, stores where a player has won a subboard, and the last board is the ones that have ended in a draw.
  uint16_t macroboards[3] {};

  // 9 means any subboard is allowed.
  size_t forced_sb {9};

  // false/0 for X to play, true/1 for O to play
  // This aligns with indexing of subboards and macroboards, as 0 means the X board and 1 means the O board.
  bool next_player {0};
  size_t moves_played {0};
};

bool is_legal (const Board board, const Move move) {
  // It is within the board
  return move.subboard < 9 && move.square < 9
    // And on the correct subboard
    && (board.forced_sb == 9 || move.subboard == board.forced_sb)
    // And the square is empty
    && ((board.subboards[move.subboard][0] | board.subboards[move.subboard][1]) & MOVE_MASKS[move.square]) == 0;
}

bool terminal (const Board board) {
  // Check if there is a win.
  for (const auto mask : WIN_MASKS) {
    for (size_t player {0}; player < 2; player++) {
      // Check for a 3 in a row overall.
      if ((mask & board.macroboards[player]) == mask) return true;
    }
  }

  // Check for a draw.
  // Happens if the board is full.
  if ((board.macroboards[0] | board.macroboards[1] | board.macroboards[2]) == FULL_BOARD) return true;

  return false;
}

// Updates the state of a subboard stored in the macroboard after a certain move is played.
void update_subboard_state (Board &board, const size_t subboard) {
  for (const auto mask : WIN_MASKS) {
    for (size_t player {0}; player < 2; player++) {
      // Win detected if everything under the mask is a 1, or in other words all squares required for a win are taken.
      if ((mask & board.subboards[subboard][player]) == mask) {
	board.macroboards[player] |= MOVE_MASKS[subboard];
	return;
      }
    }
  }
  
  // No wins detected on the subboard.
  // Check for a draw.
  // Draw occurs when all squares of a subboard have been taken.
  if ((board.subboards[subboard][0] | board.subboards[subboard][1]) == FULL_BOARD) board.macroboards[2] |= MOVE_MASKS[subboard];
}

void play_move_unsafe (Board &board, const Move move) {
  // Play the move.
  board.subboards[move.subboard][board.next_player] |= MOVE_MASKS[move.square];
  // Check if the subboard played on is now completed.
  update_subboard_state(board, move.subboard);
  // If the board played on is completed, the next move can be anywhere.
  // Otherwise it has to be on subboard corresponding to the square played on.
  if ((board.macroboards[0] | board.macroboards[1] | board.macroboards[2]) & MOVE_MASKS[move.square]) board.forced_sb = 9;
  else board.forced_sb = move.square;
  
  // Update the person playing next.
  board.next_player = !board.next_player;
  board.moves_played++;
}

Board play_move_unsafe_value (const Board board, const Move move) {
  Board res {board};

  play_move_unsafe(res, move);

  return res;
}


bool play_move (Board &board, const Move move) {
  if (is_legal(board, move)) {
    play_move_unsafe(board, move);
    return true;
  }

  return false;
}

std::string to_string (const Board &board) {
  std::array<int, 9> state {};
  for (int i {0}; i < 9; i++) {
    if (board.macroboards[0] & MOVE_MASKS[i]) state[i] = 1;
    else if (board.macroboards[1] & MOVE_MASKS[i]) state[i] = -1;
  }

  std::array<std::array<int, 9>, 9> board_array {};
  for (int b {0}; b < 9; b++) {
    for (int s {0}; s < 9; s++) {
      if (board.subboards[b][0] & MOVE_MASKS[s]) board_array[b][s] = 1;
      else if (board.subboards[b][1] & MOVE_MASKS[s]) board_array[b][s] = -1;
    }
  }
  
  return to_string(board_array, state);
}

// Returns a vector of all legal moves.
std::vector<Move> legal_moves (const Board board) {
  std::vector<Move> moves {};

  if (board.forced_sb < 9) {
    // All empty squares in the specific subboard.
    // uint16_t | uint16_t => int.
    uint16_t subboard {static_cast<uint16_t>(board.subboards[board.forced_sb][0] | board.subboards[board.forced_sb][1])};

    // For each square, check if its empty,
    for (size_t s {0}; s < 9; s++) {
      // And if so, add it to the list of valid moves.
      if (!(subboard & MOVE_MASKS[s])) moves.push_back({board.forced_sb, s});
    }
  } else {
    // Add all empty squares.
    // Same as previous, but repeated for all subboards.
    for (size_t b {0}; b < 9; b++) {
      // If the board has been completed, skip it.
      if ((board.macroboards[0] | board.macroboards[1] | board.macroboards[2]) & MOVE_MASKS[b]) continue;
      uint16_t subboard {static_cast<uint16_t>(board.subboards[b][0] | board.subboards[b][1])};
      
      for (size_t s {0}; s < 9; s++) {
	if (!(subboard & MOVE_MASKS[s])) moves.push_back({b, s});
      }
    }
  }

  return moves;
}

class Bot {
public:
  // Returns the move that it wants to play
  virtual Move operator() (const Board board) = 0;
};

class Random : public Bot {
protected:
  std::mt19937 rng;
public:
  // Template forwarding constructor.
  // Takes any arguments that mt19937 can take in its constructor, and forwards them to it.
  template<typename... Types>
  Random (Types&&... args)
    : rng {std::forward<Types>(args)...} {}
  
  Move operator() (const Board board) {
    std::vector<Move> moves {legal_moves(board)};
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
  }
};

// For later on if I get confused.
// https://www.learncpp.com/cpp-tutorial/function-pointers/
using Heuristic = std::function<int(Board)>;

class Minimax : public Bot {
protected:
  const size_t max_depth;
  Heuristic eval;
public:
  Minimax () = delete;
  
  Minimax (size_t max_depth, Heuristic eval) :
    max_depth {max_depth}, eval {eval} {}
  
  int minimax (Board board, size_t depth, int alpha, int beta) {
    if (terminal(board) || depth <= 0) return eval(board);

    const bool maximizing {board.next_player};
    
    // If it is the min player's turn, then they will try to minimize the evaluation of the board.
    // X is trying to minimize, O is trying to maximize.
    int value;
    if (!maximizing) {
      // X's turn
      // Trying to minimize, so start with the biggest possible value and find better and better eevaluations.
      value = std::numeric_limits<int>::max();
      for (const auto &move : legal_moves(board)) {
	value = std::min(value, minimax(play_move_unsafe_value(board, move), depth - 1, alpha, beta));
	beta = std::min(value, beta);
	if (beta <= alpha) break;
      }
    } else {
      // O's turn
      value = std::numeric_limits<int>::min();
      for (const auto &move : legal_moves(board)) {
	value = std::max(value, minimax(play_move_unsafe_value(board, move), depth - 1, alpha, beta));
	alpha = std::max(value, alpha);
	if (beta <= alpha) break;
      }
    }

    return value;
  }
  
  Move operator() (const Board board) {
    bool maximizing {board.next_player};
    
    auto moves = legal_moves(board);
    // Index and value of the best move.
    // If the current player is X, they are trying to minimize, so start with the biggest possible value.
    // Vice versa for O.
    std::array<int, 2> best_move {-1, maximizing ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max()};
    for (size_t i {0}; i < moves.size(); i++) {
      int value {minimax(play_move_unsafe_value(board, moves[i]), max_depth - 1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max())};

      std::cout << moves[i].subboard << moves[i].square << ": " << value << '\n';
      
      // If the current player wants to maximize, they want value to be higher than the best found so far.
      if ((maximizing && value > best_move[1]) ||
	  (!maximizing && value < best_move[1]))
	best_move = {static_cast<int>(i), value};
    }

    return moves[best_move[0]];
  }
};

// Same as minimax, except instead of playing the first best move it sees, randomly picks from the best moves.
class Minimax_random : public Minimax {
protected:
  std::mt19937 rng;
public:
  
  template <typename... Types>
  Minimax_random (size_t max_depth, Heuristic eval, Types&&... args) :
    Minimax {max_depth, eval}, rng {std::forward<Types>(args)...} {}
  
  Move operator() (const Board board) {
    bool maximizing {board.next_player};
    
    auto moves = legal_moves(board);
    // Shuffle the moves beforehand so that if there is a tie, the best one is picked randomly.
    std::shuffle(moves.begin(), moves.end(), rng);
    
    // Index and value of the best move.
    // If the current player is X, they are trying to minimize, so start with the biggest possible value.
    // Vice versa for O.
    std::array<int, 2> best_move {-1, maximizing ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max()};
    for (size_t i {0}; i < moves.size(); i++) {
      int value {minimax(play_move_unsafe_value(board, moves[i]), max_depth - 1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max())};

      std::cout << moves[i].subboard << moves[i].square << ": " << value << '\n';
      
      // If the current player wants to maximize, they want value to be higher than the best found so far.
      if ((maximizing && value > best_move[1]) ||
	  (!maximizing && value < best_move[1]))
	best_move = {static_cast<int>(i), value};
    }

    return moves[best_move[0]];
  }
};

int heur1 (const Board board) {
  // Check if there is a win
  for (const auto mask : WIN_MASKS) {
    // Player 0 (X) is minimizing.
    if ((mask & board.macroboards[0]) == mask) return -10000;
    if ((mask & board.macroboards[1]) == mask) return  10000;
  }

  // If max wins a board, +1, if min wins a board, -1.
  return std::popcount(board.macroboards[1]) - std::popcount(board.macroboards[0]);
}

int main () {
  // Sample game against a randomly playing oponent.
  const std::vector<Move> moves {
    {0, 0}, {0, 3}, {3, 2}, {2, 7}, {7, 4}, {4, 4}, {4, 1}, {1, 1}, {1, 0}, {0, 1}, {1, 3}, {3, 1}, {1, 6}, {6, 6}, {6, 1}, {6, 5}, {5, 1}, {3, 5}, {5, 8}, {8, 0}, {0, 4}, {4, 7}, {7, 2}, {2, 8}, {8, 8}, {8, 2}, {2, 6}, {6, 8}, {8, 1}, {4, 0}, {0, 8}, {8, 5}, {5, 7}, {7, 5}, {5, 6}, {6, 7}, {7, 7}, {7, 3}, {3, 4}, {4, 8}, {8, 3}, {3, 6}, {2, 2}, {2, 0}, {2, 4}
  };
  
  Board board {};
  Random random {std::random_device{}()};
  Minimax minimax2 {2, heur1};
  Minimax minimax3 {3, heur1};
  Minimax minimax4 {4, heur1};
  Minimax_random mr2 {2, heur1};
  Minimax_random mr3 {3, heur1};
  Minimax_random mr4 {4, heur1};


  std::cout << to_string(board) << "\n\n";

  Move move {};
  while (!terminal(board)) {
    std::cout << "moves played " << board.moves_played << '\n';

    if (board.next_player) {
      move = mr2(board);
      std::cout << "mr2 (maximizing) playing " << move.subboard << ' ' << move.square << '\n';
    } else {
      move = mr3(board);
      std::cout << "mr3 (minimizing) playing " << move.subboard << ' ' << move.square << '\n';
    }
    
    if (!play_move(board, move)) {
      std::cout << "Something has gone horribly wrong with trying to play the move " << move.subboard << ' ' << move.square << '\n';
      break;
    }

    std::cout << to_string(board) << "\n\n\n";
  }

  return 0;
}
