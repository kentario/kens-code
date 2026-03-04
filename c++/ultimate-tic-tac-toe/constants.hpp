#pragma once

#include <iostream>
#include <cstdint>
#include <array>
#include <vector>

constexpr size_t EMPTY_SQUARES_SIZE {12'288};

constexpr std::array<uint16_t, 8> WIN_MASKS {
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

constexpr std::array<uint16_t, 9> MOVE_MASKS {
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

// TWO_IN_A_ROW[square] => the other squares that would need to be filled in order to get a 3 in a row
constexpr size_t MAX_TWO_IN_A_ROWS_PER_SQUARE {4};
constexpr std::array<std::array<uint16_t, MAX_TWO_IN_A_ROWS_PER_SQUARE>, 9> TWO_IN_A_ROWS_ARR {{
    {0b011000000, 0b000100100, 0b000010001, 0},
    {0b101000000, 0b000010010, 0, 0},
    {0b110000000, 0b000001001, 0b000010100, 0},
    {0b100000100, 0b000011000, 0, 0},
    {0b100000001, 0b010000010, 0b001000100, 0b000101000},
    {0b001000001, 0b000110000, 0, 0},
    {0b100100000, 0b000000011, 0b001010000, 0},
    {0b010010000, 0b000000101, 0, 0},
    {0b100010000, 0b001001000, 0b000000110, 0}
  }
};
// So that I know to ignore the 0s in the above array.
constexpr std::array<size_t, 9> TWO_IN_A_ROW_COUNTS {
  3, 2, 3, 2, 4, 2, 3, 2, 3
};

constexpr uint16_t FULL_BOARD {
  0b111111111
};

constexpr uint8_t ANY_SUBBOARD {9};

constexpr double WIN {1e5};
constexpr double LOSS {-1e5};
constexpr double DRAW {0};

enum class Game_Result {
  PLAYER0_WIN,
  PLAYER1_WIN,
  DRAW
};

enum class Player : uint8_t {
  X = 0,
  O = 1
};

enum class Role : uint8_t {
  MIN = 0,
  MAX = 1
};

constexpr uint8_t to_index (const Player p) { return static_cast<uint8_t>(p); }
constexpr uint8_t to_index (const Role r) { return static_cast<uint8_t>(r); }
constexpr bool is_min (const Player p) { return p == Player::X; }
constexpr bool is_max (const Player p) { return p == Player::O; }
constexpr Player other (const Player p) { return p == Player::X ? Player::O : Player::X; }
constexpr Role other (const Role r) { return r == Role::MIN ? Role::MAX : Role::MIN; }
constexpr Player player (const Role r) { return r == Role::MIN ? Player::X : Player::O; }
constexpr Player player (const Player p) { return p; }
constexpr Role role (const Player p) { return p == Player::X ? Role::MIN : Role::MAX; }
constexpr Role role (const Role r) { return r; }
constexpr int sign (const Player p) { return p == Player::X ? -1 : 1; }
constexpr int sign (const Role r) { return r == Role::MIN ? -1 : 1; }

std::ostream& operator<< (std::ostream &os, const Player p);
std::ostream& operator<< (std::ostream &os, const Role p);
