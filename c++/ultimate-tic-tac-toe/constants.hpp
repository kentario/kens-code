#pragma once

#include <iostream>
#include <cstdint>
#include <array>

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
const std::array<std::vector<uint16_t>, 9> TWO_IN_A_ROWS {{
    {0b011000000, 0b000100100, 0b000010001},
    {0b101000000, 0b000010010},
    {0b110000000, 0b000001001, 0b000010100},
    {0b100000100, 0b000011000},
    {0b100000001, 0b010000010, 0b001000100, 0b000101000},
    {0b001000001, 0b000110000},
    {0b100100000, 0b000000011, 0b001010000},
    {0b010010000, 0b000000101},
    {0b100010000, 0b001001000, 0b000000110}
  }
};

constexpr uint16_t FULL_BOARD {
  0b111111111
};

enum class Player : uint8_t {
  X = 0,
  O = 1
};

enum class Role : uint8_t {
  MIN = 0,
  MAX = 1
};

constexpr uint8_t to_index (const Player p) {
  return static_cast<uint8_t>(p);
}

constexpr uint8_t to_index (const Role r) {
  return static_cast<uint8_t>(r);
}

constexpr bool is_min (const Player p) {
  return p == Player::X;
}

constexpr bool is_max (const Player p) {
  return p == Player::O;
}

constexpr Player other (const Player p) {
  return p == Player::X ? Player::O : Player::X;
}

constexpr Role other (const Role r) {
  return r == Role::MIN ? Role::MAX : Role::MIN;
}

constexpr Player player (const Role r) {
  return r == Role::MIN ? Player::X : Player::O;
}

constexpr Player player (const Player p) {
  return p;
}

constexpr Role role (const Player p) {
  return p == Player::X ? Role::MIN : Role::MAX;
}

constexpr Role role (const Role r) {
  return r;
}

constexpr int sign (const Player p) {
  return p == Player::X ? -1 : 1;
}

constexpr int sign (const Role r) {
  return r == Role::MIN ? -1 : 1;
}

std::ostream& operator<< (std::ostream &os, const Player p) {
  return os << (p == Player::X ? "X" : "O");
}

std::ostream& operator<< (std::ostream &os, const Role p) {
  return os << (p == Role::MIN ? "MIN" : "MAX");
}

enum class Game_Result {
  PLAYER0_WIN,
  PLAYER1_WIN,
  DRAW
};
