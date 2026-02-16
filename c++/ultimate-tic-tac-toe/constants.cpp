#include "constants.hpp"

#include <iostream>

std::ostream& operator<< (std::ostream &os, const Player p) { return os << (p == Player::X ? "X" : "O"); }
std::ostream& operator<< (std::ostream &os, const Role p) { return os << (p == Role::MIN ? "MIN" : "MAX"); }
