#pragma once

#include <unordered_map>
#include <vector>
#include <string_view>
#include <string>
#include <utility>
#include <stdexcept>

#include "token.hpp"
#include "lexer.hpp"
#include "expression.hpp"
#include "expression-factory.hpp"

/*
  https://craftinginterpreters.com/parsing-expressions.html
  
  Good resource about parsing:
  Recursive Descent Parsing by hhp3
  https://www.youtube.com/watch?v=SToUyjAsaFk
*/


namespace parser {
  using namespace token;
  using namespace expression;
  using namespace factory;

  template <Arithmetic N = double>
  Expression_Pointer<N> parse (const std::vector<Token> &tokens, int &i);

  template <Arithmetic N = double>
  Expression_Pointer<N> parse (const std::vector<Token> &tokens);

  // Parse a string of a number into some arithmetic type N.
  // Uses a bunch of messy template specialization.
  template <Arithmetic N>
  N parse_string (const std::string &s);

  template <>
  int parse_string<int> (const std::string &s) {
    return std::stoi(s);
  }

  template <>
  long parse_string<long> (const std::string &s) {
    return std::stol(s);
  }

  template <>
  float parse_string<float> (const std::string &s) {
    return std::stof(s);
  }

  template <>
  double parse_string<double> (const std::string &s) {
    return std::stod(s);
  }

  template <>
  long double parse_string<long double> (const std::string &s) {
    return std::stold(s);
  }
  
  // i is the index of the current token being evaluated.
  // If the current token is one of the types specified, then returns true and consumes it (meaning i is incremented past it).
  // Otherwise returns false and does nothing.
  bool match (const std::vector<Token_Type> &types, const std::vector<Token> &tokens, int &i) {
    for (const auto &type : types) {
      if (tokens[i].type == type && tokens[i].type != Token_Type::END_OF_FILE) {
	i++;
	return true;
      }
    }
    
    return false;
  }
  
  template <Arithmetic N>
  Expression_Pointer<N> primary (const std::vector<Token> &tokens, int &i) {
    if (match({Token_Type::NUMBER}, tokens, i)) return std::make_unique<operands::Number<N>>(parse_string<N>(std::string(tokens[i - 1].value)));
    if (match({Token_Type::VARIABLE}, tokens, i)) return std::make_unique<operands::Variable<N>>(std::string(tokens[i - 1].value));

    // If there is an open parenthesis, then recursively collapse everything after into an expression, then look for the corresponding close parenthesis after that.
    if (match({Token_Type::OPEN_PARENTHESIS}, tokens, i)) {
      auto expr = parse(tokens, i);
      // If there is a close parenthesis, then consume it.
      if (match({Token_Type::CLOSE_PARENTHESIS}, tokens, i)) {
	return expr;
      } else {
	throw std::runtime_error {"Expected close parenthesis at " + std::to_string(i) + "\n"};
      }
    }

    throw std::runtime_error {"Somehow got to the end of primary() at " + std::to_string(i) + "\n"};
  }
  
  template <Arithmetic N>
  Expression_Pointer<N> unary (const std::vector<Token> &tokens, int &i) {
    // If there is a unary operator, then the stuff to its right get collapsed into a unary, then collapse this and the stuff to the right into the final expression
    if (match({Token_Type::SUBTRACTION, Token_Type::SQUARE_ROOT}, tokens, i)) {
      Token_Type operation {tokens[i - 1].type};
      auto right = unary<N>(tokens, i);
      return make_unary_operator(operation, std::move(right));
    }

    // Otherwise, if there was no unary operator, then the stuff to the right must be even higher precedence, so collapse that stuff.
    return primary<N>(tokens, i);
  }

  template <Arithmetic N>
  Expression_Pointer<N> power (const std::vector<Token> &tokens, int &i) {
    auto expr = unary<N>(tokens, i);

    if (match({Token_Type::POWER}, tokens, i)) {
      // If there is a '^' token, then there is another potential exponent
      auto right = power<N>(tokens, i);
      return make_binary_operator<operators::Power>(std::move(expr), std::move(right));
    }

    return expr;
  }
  
  template <Arithmetic N>
  Expression_Pointer<N> factor (const std::vector<Token> &tokens, int &i) {
    // Uses the same logic as term.
    auto expr = power<N>(tokens, i);

    while (match({Token_Type::MULTIPLICATION, Token_Type::DIVISION}, tokens, i)) {
      Token_Type operation {tokens[i - 1].type};
      auto right = power<N>(tokens, i);
      expr = make_binary_operator(operation, std::move(expr), std::move(right));
    }

    return expr;
  }
  
  template <Arithmetic N>
  Expression_Pointer<N> term (const std::vector<Token> &tokens, int &i) {
    // Convert everything we can of the tokens into stuff including everythign up to * and /.
    /*
      3 * 5 / 2 + x / 10 * 2 - 3
        -> expr + x / 10 * 2 - 3
     */
    auto expr = factor<N>(tokens, i);

    // While the thing after factor is a term operation ('+' or '-'), continue adding things onto the end
    while (match({Token_Type::ADDITION, Token_Type::SUBTRACTION}, tokens, i)) {
      // expr + x / 10 * 2 - 3
      const Token_Type operation {tokens[i - 1].type};
      // expr operation x / 10 * 2 - 3
      auto right = factor<N>(tokens, i);
      // expr operation right - 3
      expr = make_binary_operator(operation, std::move(expr), std::move(right));
      // expr - 3
      // repeat ...
    } // expr

    return expr;
  }
  
  template <Arithmetic N = double>
  Expression_Pointer<N> parse (const std::vector<Token> &tokens, int &i) {
    // Converts the tokens into a term, meaning everything up to + and -.
    return term<N>(tokens, i);
  }

  template <Arithmetic N = double>
  inline Expression_Pointer<N> parse (const std::vector<Token> &tokens) {
    int i {0};
    return parse(tokens, i);
  }

} // namespace parser
