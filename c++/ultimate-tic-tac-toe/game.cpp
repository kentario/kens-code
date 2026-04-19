#include "game.hpp"

#include <filesystem>
#include <iostream>
#include <cstdint>
#include <string>
#include <array>
#include <vector>
#include <fstream>
#include <exception>
#include <span>
#include <string_view>

#include "constants.hpp"

bool operator== (const Move a, const Move b) {
  return a.subboard == b.subboard && a.square == b.square;
}

std::string to_string (const Move move) {
  return "(" + std::to_string(move.subboard) + " " + std::to_string(move.square) + ")";
}

std::ostream& operator<< (std::ostream &os, const Move move) {
  return os << to_string(move);
}

Squares_List& Squares_List::push_back (const uint8_t m) {
  squares[size++] = m;

  return *this;
}

uint8_t* Squares_List::begin () {return squares.data();}
uint8_t* Squares_List::end () {return squares.data() + size;}

std::array<Squares_List, 512> Board::empty_squares {};

Player Board::next_player () const {
  return static_cast<Player>(moves_played & 1);
}

bool Board::terminal () const {
  // Check if there is a win.
  for (const auto mask : WIN_MASKS) {
    for (size_t player {0}; player < 2; player++) {
      // Check for a 3 in a row overall.
      if ((mask & macroboards[player]) == mask) return true;
    }
  }

  // Check for a draw.
  // Happens if the board is full.
  if ((macroboards[to_index(Player::X)] | macroboards[to_index(Player::O)] | macroboards[2]) == FULL_BOARD) return true;

  return false;
}

bool Board::board_completed (const size_t subboard) const {
  return ((macroboards[to_index(Player::X)] | macroboards[to_index(Player::O)] | macroboards[2]) & MOVE_MASKS[subboard]);
}

bool Board::is_legal (const Move move) const {
  // It is within the board
  return move.subboard < 9 && move.square < 9
    // And on the correct subboard
    && (forced_sb == ANY_SUBBOARD || move.subboard == forced_sb)
    // And the square is empty
    && ((subboards[move.subboard][to_index(Player::X)] | subboards[move.subboard][to_index(Player::O)]) & MOVE_MASKS[move.square]) == 0;
}

// Updates the state of a subboard stored in the macroboard after a certain move is played.
void Board::update_subboard_state (const size_t subboard) {
  for (const auto mask : WIN_MASKS) {
    // Win detected if everything under the mask is a 1, or in other words all squares required for a win are taken.
    // Only the player who just played could have won a board.
    auto player = to_index(other(next_player()));
    if ((mask & subboards[subboard][player]) == mask) {
      macroboards[player] |= MOVE_MASKS[subboard];
      return;
    }
  }
  
  // No wins detected on the subboard.
  // Check for a draw.
  // Draw occurs when all squares of a subboard have been taken.
  if ((subboards[subboard][to_index(Player::X)] | subboards[subboard][to_index(Player::O)]) == FULL_BOARD) {
    macroboards[2] |= MOVE_MASKS[subboard];
  }
}

void Board::play_move_unsafe (const Move move) {
  // Play the move.
  subboards[move.subboard][to_index(next_player())] |= MOVE_MASKS[move.square];
  // Check if the subboard played on is now completed.
  update_subboard_state(move.subboard);
  // If the board played on is completed, the next move can be anywhere.
  // Otherwise it has to be on subboard corresponding to the square played on.
  if (board_completed(move.square)) forced_sb = ANY_SUBBOARD;
  else forced_sb = move.square;

  moves_played_vector[moves_played] = move;
  moves_played++;
}

Board Board::play_move_unsafe_value (const Move move) const {
  Board result {*this};
  result.play_move_unsafe(move);
    
  return result;
}

// Returns whether the move succeeded.
bool Board::play_move (const Move move) {
  if (is_legal(move)) {
    play_move_unsafe(move);
    return true;
  }

  return false;
}

void Board::undo_move () {
  if (moves_played <= 0) return;

  Player player_played {other(next_player())};
  Move move_undone {moves_played_vector[moves_played - 1]};
  // To erase a certain bit of a subboard  (this resets the square state),
  // AND the subboard with all 1s except a 0 at the index of the square
  // to get this, do ~MOVE_MASK
  subboards[move_undone.subboard][to_index(player_played)] &= ~MOVE_MASKS[move_undone.square];
  if (board_completed(move_undone.subboard)) {
    // If the board played on is completed, then it should be uncompleted.
    // Same logic as squares with erasing macroboard (subboard states).
    macroboards[to_index(player_played)] &= ~MOVE_MASKS[move_undone.subboard];
    // Always erase draws just in case.
    macroboards[2] &= ~MOVE_MASKS[move_undone.subboard];
  }
  moves_played--;

  if (!moves_played &&
      board_completed(moves_played_vector[moves_played - 1].square)) forced_sb = ANY_SUBBOARD;
  else forced_sb = moves_played_vector[moves_played - 1].square;
}

std::vector<Move> Board::legal_moves () const {
  std::vector<Move> moves {};

  if (forced_sb == ANY_SUBBOARD) {
    // Iterate over each non-completed subboard.
    auto empty_subboards = empty_squares[macroboards[0] | macroboards[1] | macroboards[2]];
    // Maximum number of possible moves.
    moves.reserve(empty_subboards.size * 9);
    for (const uint8_t board_i : empty_subboards) {
      const auto x_subboard = subboards[board_i][to_index(Player::X)];
      const auto o_subboard = subboards[board_i][to_index(Player::O)];
      const auto &move_list = empty_squares[x_subboard | o_subboard];

      for (size_t i {0}; i < move_list.size; i++) {
	const uint8_t square_i {move_list.squares[i]};
	moves.push_back({board_i, square_i});
      }
    }
  } else {
    const auto x_subboard = subboards[forced_sb][to_index(Player::X)];
    const auto o_subboard = subboards[forced_sb][to_index(Player::O)];
    const auto &move_list = empty_squares[x_subboard | o_subboard];

    moves.reserve(empty_squares[x_subboard | o_subboard].size);

    for (size_t i {0}; i < move_list.size; i++) {
      const uint8_t square_i {move_list.squares[i]};
      moves.push_back({forced_sb, square_i});
    }
  }

  return moves;
}

// The number of empty squares in a specific subboard.
size_t Board::count_empty_squares (const size_t subboard) const {
  // Bitboard for all taken squares in the subboard.
  const uint16_t full_subboard {
    static_cast<uint16_t>
    (subboards[subboard][to_index(Player::X)] |
     subboards[subboard][to_index(Player::O)])
  };

  return empty_squares[full_subboard].size;
}
  
// Total empty squares in the entire board.
size_t Board::count_total_empty_squares () const {
  size_t count {0};

  for (size_t b {0}; b < 9; b++) {
    // If the board has been completed, skip it.
    if (board_completed(b)) continue;
      
    count += count_empty_squares(b);
  }

  return count;
}
  
int Board::count_legal_moves () const {
  if (forced_sb < ANY_SUBBOARD) {
    return count_empty_squares(forced_sb);
  } else {
    return count_total_empty_squares();
  }
}

void Board::pre_generate_legal_moves (const bool overwrite) {
  if (!overwrite && std::filesystem::exists("pre-generated-moves")) {
    std::cout << "loading moves\n";
      
    // If the file already exists, it should just be loaded.
    std::ifstream in {"pre-generated-moves", std::ios::binary};

    if (!in) throw std::runtime_error {"pre_generate_legal_moves failed to open file pre-generated-moves"};

    in.read(reinterpret_cast<char*>(&empty_squares), EMPTY_SQUARES_SIZE);
    return;
  }

  // The array takes a 9 bit number with a 1 at every occupied square.
  // A 9 bit number has 2^9 = 512 possible combinations.
  empty_squares = std::array<Squares_List, 512> {};
    
  std::cout << "generating moves\n";
    
  for (uint16_t subboard {0}; subboard <= FULL_BOARD; subboard++) {
    // For each square, check if its empty,
    for (uint8_t s {0}; s < 9; s++) {
      // And if so, add it to the list of valid moves.
      if (!(subboard & MOVE_MASKS[s])) {
	empty_squares[subboard].push_back(s);
      }
    }
  }

  // Store the moves in a file for later use.
  std::ofstream out {"pre-generated-moves", std::ios::binary};
  if (!out) throw std::runtime_error {"pre_generate_legal_moves failed to open file pre-generated-moves"};
    
  out.write(reinterpret_cast<const char*>(&empty_squares), sizeof(empty_squares));
  //    std::cout << sizeof(empty_squares) << '\n';
}

void save_positions (const std::string &filename, const Board board[], const size_t count, const bool append) {
  std::ofstream out {filename, std::ios::binary | (append ? std::ios::app : std::ios::trunc)};
  if (!out) throw std::runtime_error {"save_positions failed to open file " + filename};

  out.write(reinterpret_cast<const char*>(board), sizeof(Board) * count);
}

void update_translate_index (std::span<size_t> translate_index, const size_t inserted_location, std::string_view str) {
  for (size_t i {inserted_location + 1}; i < translate_index.size(); i++) {
    translate_index[i] += str.size();
  }
}

std::vector<Board> load_positions (const std::string &filename, const size_t count) {
  // Starting at the end to determine file size.
  std::ifstream in {filename, std::ios::binary | std::ios::ate};
  if (!in) throw std::runtime_error {"load_positions failed to open file " + filename};

  std::streamsize size {in.tellg()};
  // Only boards are stored in the file, so the file size should be a multiple of the board size.
  if (size % sizeof(Board) != 0) throw std::runtime_error {"Corrupt board file"};

  // Go back to beginning to read data.
  in.seekg(0);

  // If there aren't enough boards stored.
  if (count > size/sizeof(Board)) throw std::runtime_error {"Not enough stored board positions"};
  
  std::vector<Board> boards(count);
  
  in.read(reinterpret_cast<char*>(boards.data()), sizeof(Board) * count);

  return boards;
}

std::string to_string (const Board &board) {
  // Convert bitboard representation into array of boards.
  // Winner of each subboard.
  std::array<int, 9> state {};
  for (int i {0}; i < 9; i++) {
    if (board.macroboards[0] & MOVE_MASKS[i]) state[i] = 1;
    else if (board.macroboards[1] & MOVE_MASKS[i]) state[i] = -1;
  }

  // The pieces in the entire board.
  // board_array[subboard][square]
  std::array<std::array<int, 9>, 9> board_array {};
  for (int b {0}; b < 9; b++) {
    for (int s {0}; s < 9; s++) {
      if (board.subboards[b][0] & MOVE_MASKS[s]) board_array[b][s] = 1;
      else if (board.subboards[b][1] & MOVE_MASKS[s]) board_array[b][s] = -1;
    }
  }

  // Convert array of boards into string.
  std::string res {};

  for (int row {0}; row < 9; row++) {
    for (int col {0}; col < 9; col++) {
      const int subboard {(col/3) + (row/3) * 3};
      const int square {(col % 3) + (row % 3) * 3};

      const int v {board_array[subboard][square]};

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

  // For each winning subboard, make a big version of the shape on top.
  const std::string big_x {"\\   /                        \\ /                          X                          / \\                        /   \\"};
  const std::string big_o {" /^\\                        |   |                       |   |                       |   |                        \\_/ "};
  size_t top_left {0};
  for (size_t i {0}; i < 9; i++) {
    // The state of the current subboard.
    const int s {state[i]};
    
    switch (s) {
    case 0: // Uncompleted/draw
      break;
    case 1: // X won
      for (size_t j {0}; j < big_x.size(); j++) {
	if (j % 28 < 5) {
	  res[top_left + j] = big_x[j];
	}
      }
      break;
    case -1: // O won
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

  // Highlight the forced subboard.
  /*
    For testing different colors
    for i in {30..37}; do echo -e "\033[1;$i""mcolorful text\033[0m"; done
  */
  const std::string highlight_start {"\033[1;36m"};
  const std::string highlight_end {"\033[0m"};
  //  const std::string highlight_start {"s"};
  //  const std::string highlight_end {"e"};

  Squares_List subboards_todo {};
  if (board.forced_sb == 9) {
    subboards_todo = Board::empty_squares[board.macroboards[0] | board.macroboards[1] | board.macroboards[2]];
  } else {
    subboards_todo.push_back(board.forced_sb);
  }

  constexpr size_t original_res_size {615};
  // + 1 because insertions can happen at index 0 but also at index size() to go at the very end.
  std::array<size_t, original_res_size + 1> translate_index {};
  for (size_t i {0}; i < translate_index.size(); i++) {
    translate_index[i] = i;
  }
  
  // For each empty subboard,
  for (const uint8_t subboard_i : subboards_todo) {
    // Horizontal offset
    top_left = 11 * (subboard_i % 3);
    // If the subboard is in a lower row, the borders between rows must be skipped.
    top_left += (subboard_i/3) * 238;

    const size_t row_size {28};

    for (int row_offset {0}; row_offset <= 4; row_offset++) {
      size_t start_position_before {
	// Start
	top_left
	// + rows alrady done
	+ row_offset * row_size
      };
      res.insert(translate_index[start_position_before], highlight_start);
      update_translate_index(translate_index, start_position_before, highlight_start);
      
      size_t end_position_before {
	// Start
	top_left
	// + rows already done
	+ row_offset * row_size
	// + distance between start and end of highlight.
	+ 5
      };
      res.insert(translate_index[end_position_before], highlight_end);
      update_translate_index(translate_index, end_position_before, highlight_end);
    }
  }

  return res;
}

std::ostream& operator<< (std::ostream &os, const Board &board) {
  return os << to_string(board);
}
