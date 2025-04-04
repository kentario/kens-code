#pragma once

#include <vector>
#include <array>
#include <string_view>
#include <iostream>

namespace token {

  enum class Token_Type {
    NUMBER,
    VARIABLE,

    OPEN_PARENTHESIS,
    CLOSE_PARENTHESIS,

    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,

    SQUARE_ROOT,

    END_OF_FILE
  };

  struct Token {
    const Token_Type type;
    const std::string_view value;
    
    Token (const Token_Type &type, const std::string_view value) :
      type {type}, value {value} {}

    Token (const Token_Type &type, const char* value) :
      type {type}, value {value} {}

    Token (const Token_Type &type) :
      type {type}, value {""} {}
  };

  std::ostream& operator<< (std::ostream &os, const Token &token) {
    std::string type_string {};
    switch (token.type) {
    case Token_Type::NUMBER:
      type_string = "Number";
      break;
    case Token_Type::VARIABLE:
      type_string = "Variable";
      break;
    case Token_Type::ADDITION:
      type_string = "Addition";
      break;
    case Token_Type::SUBTRACTION:
      type_string = "Subtraction";
      break;
    case Token_Type::MULTIPLICATION:
      type_string = "Multiplication";
      break;
    case Token_Type::DIVISION:
      type_string = "Division";
      break;
    case Token_Type::OPEN_PARENTHESIS:
      type_string = "Open Parenthesis";
      break;
    case Token_Type::CLOSE_PARENTHESIS:
      type_string = "Close Parenthesis";
      break;
    case Token_Type::SQUARE_ROOT:
      type_string = "Square Root";
      break;
    case Token_Type::END_OF_FILE:
      type_string = "EOF";
      break;
    }

    os << '{' << type_string << ", " << token.value << '}';

    return os;
  }

} // namespace token
