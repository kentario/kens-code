#pragma once

#include <unordered_map>
#include <string_view>
#include <utility>
#include <stdexcept>

#include <vector>

#include "expression.hpp"
#include "token.hpp"
#include "lexer.hpp"

/*
  Good resource about parsing:
  Recursive Descent Parsing by hhp3
  https://www.youtube.com/watch?v=SToUyjAsaFk
*/

namespace parser {

  using namespace token;
  
  template <expression::Arithmetic N = double>
  expression::Expression_Pointer<N> parse (const std::vector<Token> &tokens) {
    return {};
  }

} // namespace parser
