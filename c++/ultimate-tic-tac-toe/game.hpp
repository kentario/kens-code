#pragma once

#include <iostream>
#include <cstdint>
#include <string>
#include <string_view>
#include <array>
#include <vector>
#include <fstream>
#include <exception>

#include "constants.hpp"

struct Move {
  size_t subboard {};
  size_t square {};
};

std::string to_string (const Move move) {
  return std::to_string(move.subboard) + " " + std::to_string(move.square);
}

std::ostream& operator<< (std::ostream &os, const Move move) {
  return os << '(' << move.subboard << ' ' << move.square << ')';
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

  Player next_player () const {
    return static_cast<Player>(moves_played & 1);
  }
};
static_assert(sizeof(Board) == 2 * 9 * 2 + 2 * 3 + 1 + 1);

bool is_legal (const Board board, const Move move) {
  // It is within the board
  return move.subboard < 9 && move.square < 9
    // And on the correct subboard
    && (board.forced_sb == ANY_SUBBOARD || move.subboard == board.forced_sb)
    // And the square is empty
    && ((board.subboards[move.subboard][to_index(Player::X)] | board.subboards[move.subboard][to_index(Player::O)]) & MOVE_MASKS[move.square]) == 0;
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
  if ((board.macroboards[to_index(Player::X)] | board.macroboards[to_index(Player::O)] | board.macroboards[2]) == FULL_BOARD) return true;

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
  if ((board.subboards[subboard][to_index(Player::X)] | board.subboards[subboard][to_index(Player::O)]) == FULL_BOARD) board.macroboards[2] |= MOVE_MASKS[subboard];
}

void play_move_unsafe (Board &board, const Move move) {
  // Play the move.
  board.subboards[move.subboard][to_index(board.next_player())] |= MOVE_MASKS[move.square];
  // Check if the subboard played on is now completed.
  update_subboard_state(board, move.subboard);
  // If the board played on is completed, the next move can be anywhere.
  // Otherwise it has to be on subboard corresponding to the square played on.
  if ((board.macroboards[to_index(Player::X)] | board.macroboards[to_index(Player::O)] | board.macroboards[2]) & MOVE_MASKS[move.square])
    board.forced_sb = ANY_SUBBOARD;
  else board.forced_sb = move.square;
  
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

// Returns a vector of all legal moves.
std::vector<Move> legal_moves (const Board board) {
  std::vector<Move> moves {};

  // If a specific subboard is being forced,
  if (board.forced_sb < ANY_SUBBOARD) {
    // then add all empty squares that subboard.
    // uint16_t | uint16_t => int.
    uint16_t subboard {static_cast<uint16_t>(board.subboards[board.forced_sb][to_index(Player::X)] | board.subboards[board.forced_sb][to_index(Player::O)])};

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
      if ((board.macroboards[to_index(Player::X)] | board.macroboards[to_index(Player::O)] | board.macroboards[2]) & MOVE_MASKS[b]) continue;
      
      uint16_t subboard {
	static_cast<uint16_t>
	(board.subboards[b][to_index(Player::X)] | board.subboards[b][to_index(Player::O)])
      };
      
      for (size_t s {0}; s < 9; s++) {
	if (!(subboard & MOVE_MASKS[s])) moves.push_back({b, s});
      }
    }
  }

  return moves;
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

void save_positions (const std::string &filename, const Board board[], const size_t length, const bool append) {
  std::ofstream out {filename, std::ios::binary | (append ? std::ios::app : std::ios::trunc)};
  if (!out) throw std::runtime_error {"Failed to open file"};

  out.write(reinterpret_cast<const char*>(board), sizeof(Board) * length);
}

std::vector<Board> load_positions (const std::string &filename) {
  std::vector<Board> boards {};
  
  std::ifstream in {filename, std::ios::binary};
  if (!in) throw std::runtime_error {"Failed to open file"};


  return boards;
}
