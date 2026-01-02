#pragma once

#include <filesystem>
#include <iostream>
#include <cstdint>
#include <string>
#include <string_view>
#include <array>
#include <vector>
#include <fstream>
#include <exception>
#include <functional>
#include <bit>

#include "constants.hpp"

struct Move {
  size_t subboard {};
  size_t square {};
};

bool operator== (const Move a, const Move b) {
  return a.subboard == b.subboard && a.square == b.square;
}

std::string to_string (const Move move) {
  return std::to_string(move.subboard) + " " + std::to_string(move.square);
}

std::ostream& operator<< (std::ostream &os, const Move move) {
  return os << '(' << move.subboard << ' ' << move.square << ')';
}

// Counts the number of winning moves by the first player on some tic-tac-toe board.
// Could be a macroboard, or could be a subboard.
size_t count_winning_moves (const uint16_t a, const uint16_t b) {
  size_t count {0};
  // The board of non-empty squares:
  const uint16_t filled {static_cast<uint16_t>(a | b)};
  // For each empty square, count the number of two in a rows that correspond to it.
  for (int s {0}; s < 9; s++) {
    // If the square is taken, skip it.
    if (filled & MOVE_MASKS[s]) continue;
    // Otherwise, check if the two in a rows are being satisfied.
    for (const uint16_t two_in_a_row : TWO_IN_A_ROWS[s]) {
      if ((two_in_a_row & a) == two_in_a_row) count++;
    }
  }

  return count;
}

static constexpr uint8_t ANY_SUBBOARD {9};
struct Board {
  // For both, [0] => X, [1] => O, and for macroboards, [2] => draw
  // 9 boards, left to right top to bottom, and 1 for each player.
  // 0b100000000 is just the top left cell
  uint16_t subboards[9][2] {};
  // 1 overall board for each player, stores where a player has won a subboard, and the last board is the ones that have ended in a draw.
  uint16_t macroboards[3] {};

  uint8_t forced_sb {ANY_SUBBOARD};

  // When moves_played is even, X will play next.
  // The last bit of moves played will be 0 = X when moves played is even.
  // To get the next player, just do moves_played & 1.
  uint8_t moves_played {0};

  static std::array<std::array<std::vector<uint8_t>, 512>, 512> moves_table;
  
  Player next_player () const {
    return static_cast<Player>(moves_played & 1);
  }

  bool terminal () const {
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

  bool board_completed (const size_t subboard) const {
    return ((macroboards[to_index(Player::X)] | macroboards[to_index(Player::O)] | macroboards[2]) & MOVE_MASKS[subboard]);
  }

  bool is_legal (const Move move) const {
    // It is within the board
    return move.subboard < 9 && move.square < 9
      // And on the correct subboard
      && (forced_sb == ANY_SUBBOARD || move.subboard == forced_sb)
      // And the square is empty
      && ((subboards[move.subboard][to_index(Player::X)] | subboards[move.subboard][to_index(Player::O)]) & MOVE_MASKS[move.square]) == 0;
  }

  // Updates the state of a subboard stored in the macroboard after a certain move is played.
  void update_subboard_state (const size_t subboard) {
    for (const auto mask : WIN_MASKS) {
      for (size_t player {0}; player < 2; player++) {
	// Win detected if everything under the mask is a 1, or in other words all squares required for a win are taken.
	if ((mask & subboards[subboard][player]) == mask) {
	  macroboards[player] |= MOVE_MASKS[subboard];
	  return;
	}
      }
    }
  
    // No wins detected on the subboard.
    // Check for a draw.
    // Draw occurs when all squares of a subboard have been taken.
    if ((subboards[subboard][to_index(Player::X)] | subboards[subboard][to_index(Player::O)]) == FULL_BOARD) macroboards[2] |= MOVE_MASKS[subboard];
  }

  void play_move_unsafe (const Move move) {
    // Play the move.
    subboards[move.subboard][to_index(next_player())] |= MOVE_MASKS[move.square];
    // Check if the subboard played on is now completed.
    update_subboard_state(move.subboard);
    // If the board played on is completed, the next move can be anywhere.
    // Otherwise it has to be on subboard corresponding to the square played on.
    if (board_completed(move.square)) forced_sb = ANY_SUBBOARD;
    else forced_sb = move.square;
  
    moves_played++;
  }

  Board play_move_unsafe_value (const Move move) const {
    Board result {*this};
    result.play_move_unsafe(move);
    
    return result;
  }

  // Returns whether the move succeeded.
  bool play_move (const Move move) {
    if (is_legal(move)) {
      play_move_unsafe(move);
      return true;
    }

    return false;
  }

  // Returns a vector of all legal moves.
  std::vector<Move> legal_moves () const {
    std::vector<Move> moves {};

    // If a specific subboard is being forced,
    if (forced_sb < ANY_SUBBOARD) {
      // then add all empty squares that subboard.
      
      // uint16_t | uint16_t => int.
      const uint16_t subboard {
	static_cast<uint16_t>
	(subboards[forced_sb][to_index(Player::X)] |
	 subboards[forced_sb][to_index(Player::O)])
      };

      // For each square, check if its empty,
      for (size_t s {0}; s < 9; s++) {
	// And if so, add it to the list of valid moves.
	if (!(subboard & MOVE_MASKS[s])) moves.push_back({forced_sb, s});
      }
    } else {
      // Add all empty squares.
      // Same as previous, but repeated for all subboards.
      for (size_t b {0}; b < 9; b++) {
	if (board_completed(b)) continue;
      
	const uint16_t subboard {
	  static_cast<uint16_t>
	  (subboards[b][to_index(Player::X)] |
	   subboards[b][to_index(Player::O)])
	};
      
	for (size_t s {0}; s < 9; s++) {
	  if (!(subboard & MOVE_MASKS[s])) moves.push_back({b, s});
	}
      }
    }

    return moves;
  }

  std::vector<Move> legal_moves_new () const {
    std::vector<Move> moves {};

    if (forced_sb == ANY_SUBBOARD) {
      for (size_t b {0}; b < 9; b++) {
	if (board_completed(b)) continue;
	
	for (const auto s : moves_table[subboards[b][to_index(Player::X)]][subboards[b][to_index(Player::O)]]) {
	  moves.push_back({forced_sb, s});
	}
      }
    } else {
      for (const auto s : moves_table[subboards[forced_sb][to_index(Player::X)]][subboards[forced_sb][to_index(Player::O)]]) {
	moves.push_back({forced_sb, s});
      }
    }

    return moves;
  }

  // The number of empty squares in a specific subboard.
  size_t count_empty_squares (const size_t subboard) const {
    // Bitboard for all taken squares in the subboard.
    const uint16_t full_subboard {
      static_cast<uint16_t>
      (subboards[subboard][to_index(Player::X)] |
       subboards[subboard][to_index(Player::O)])
    };
    // Count number of 0s.
    // There are 9 bits being used, and the number of 0s would be 9 minus the number of 1s.
    return 9 - std::popcount(full_subboard);
  }
  
  // Total empty squares in the entire board.
  size_t count_total_empty_squares () const {
    size_t count {0};
    
    for (size_t b {0}; b < 9; b++) {
      // If the board has been completed, skip it.
      if (board_completed(b)) continue;
      
      count += count_empty_squares(b);
    }

    return count;
  }
  
  int count_legal_moves () const {
    if (forced_sb < ANY_SUBBOARD) {
      return count_empty_squares(forced_sb);
    } else {
      return count_total_empty_squares();
    }
  }

  static void pre_generate_legal_moves (const bool overwrite) {
    if (!overwrite && std::filesystem::exists("pre-generated-moves")) {
      std::cout << "loading moves\n";
      
      // If the file already exists, it should just be loaded.
      std::ifstream in {"pre-generated-moves", std::ios::binary};

      if (!in) throw std::runtime_error {"Failed to open file pre-generated-moves"};

      in.read(reinterpret_cast<char*>(&moves_table), MOVES_TABLE_SIZE);
      return;
    }
    
    // The array takes a subboard (the two different colors as input to the 2d array), and outputs a vector of all the empty positions.
    // A subboard for a specific color uses 9 bits, so there are 2^9 = 512 different combinations.
    std::array<std::array<std::vector<uint8_t>, 512>, 512> moves {};
    
    std::cout << "generating moves\n";
    
    for (uint16_t subboard_a {0}; subboard_a <= FULL_BOARD; subboard_a++) {
      for (uint16_t subboard_b {0}; subboard_b <= FULL_BOARD; subboard_b++) {
	// Has a 1 at every taken position on the combined subboard.
	const uint16_t combined_subboard {static_cast<uint16_t>(subboard_a|subboard_b)};

	// For each square, check if its empty,
	for (uint8_t s {0}; s < 9; s++) {
	  // And if so, add it to the list of valid moves.
	  if (!(combined_subboard & MOVE_MASKS[s])) moves[subboard_a][subboard_b].push_back(s);
	}
      }
    }

    // Store the moves in a file for later use.
    std::ofstream out {"pre-generated-moves", std::ios::binary};
    if (!out) throw std::runtime_error {"Failed to open file"};
    
    out.write(reinterpret_cast<const char*>(&moves), sizeof(moves));
    std::cout << sizeof(moves) << '\n';
  }
};
std::array<std::array<std::vector<uint8_t>, 512>, 512> Board::moves_table {};
static_assert(sizeof(Board) == 2 * 9 * 2 + 2 * 3 + 1 + 1);

void save_positions (const std::string &filename, const Board board[], const size_t count, const bool append) {
  std::ofstream out {filename, std::ios::binary | (append ? std::ios::app : std::ios::trunc)};
  if (!out) throw std::runtime_error {"Failed to open file"};

  out.write(reinterpret_cast<const char*>(board), sizeof(Board) * count);
}

std::vector<Board> load_positions (const std::string &filename, const size_t count) {
  // Starting at the end to determine file size.
  std::ifstream in {filename, std::ios::binary | std::ios::ate};
  if (!in) throw std::runtime_error {"Failed to open file"};

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

  // Convert array of boards into string.
  std::string res {};

  for (int row {0}; row < 9; row++) {
    for (int col {0}; col < 9; col++) {
      const int v {board_array[(col/3) + (row/3) * 3][(col % 3) + (row % 3) * 3]};
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

std::ostream& operator<< (std::ostream &os, const Board board) {
  return os << to_string(board);
}
