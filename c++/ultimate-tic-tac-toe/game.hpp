#pragma once

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

struct Move {
  uint8_t subboard {};
  uint8_t square {};
};

bool operator== (const Move a, const Move b);
std::string to_string (const Move move);
std::ostream& operator<< (std::ostream &os, const Move move);

struct Squares_List {
  size_t size {};
  std::array<uint8_t, 9> squares;

  Squares_List& push_back (const uint8_t m);

  uint8_t* begin ();
  uint8_t* end ();
};

struct Board {
  std::array<Move, 81> moves_played_vector {};
  
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

  static std::array<Squares_List, 512> empty_squares;
  
  Player next_player () const;
  bool terminal () const;
  bool board_completed (const size_t subboard) const;
  bool is_legal (const Move move) const;
  // Updates the state of a subboard stored in the macroboard after a certain move is played.
  
  void update_subboard_state (const size_t subboard);
  void play_move_unsafe (const Move move);
  Board play_move_unsafe_value (const Move move) const;
  // Returns whether the move succeeded.
  bool play_move (const Move move);
  void undo_move ();
  
  std::vector<Move> legal_moves () const;
  
  // The number of empty squares in a specific subboard.
  size_t count_empty_squares (const size_t subboard) const;
  // Total empty squares in the entire board. Skips completed boards.
  size_t count_total_empty_squares () const;
  int count_legal_moves () const;

  static void pre_generate_legal_moves (const bool overwrite);
};
static_assert(sizeof(Board) == sizeof(std::array<Move, 81>) + 2 * 9 * 2 + 2 * 3 + 1 + 1);
// moves_played_vector
// subboards 2x9 array, 2 bytes/uint16_t
// macroboards 3 array, 2 bytes/uint16_t
// forced_sb 1 byte/uint8_t
// moves_played 1 byte/uint8_t

void save_positions (const std::string &filename, const Board board[], const size_t count, const bool append);
std::vector<Board> load_positions (const std::string &filename, const size_t count);

void update_translate_index (std::span<size_t> translate_index, const size_t inserted_location, std::string_view str);
std::string to_string (const Board &board);
std::ostream& operator<< (std::ostream &os, const Board &board);
