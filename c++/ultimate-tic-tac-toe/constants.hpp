#pragma once

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

constexpr int sign (const Player p) {
  return p == Player::X ? -1 : 1;
}

constexpr int sign (const Role r) {
  return r == Role::MIN ? -1 : 1;
}
