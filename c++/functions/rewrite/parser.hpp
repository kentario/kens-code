#pragma once

#include <unordered_map>
#include <string_view>
#include <utility>
#include <stdexcept>

#include <vector>

#include "expression.hpp"
#include "token.hpp"

/*
  Good resource about parsing:
  Recursive Descent Parsing by hhp3
  https://www.youtube.com/watch?v=SToUyjAsaFk
*/

namespace lexer {

  using namespace token;

  const std::unordered_map<std::string_view, Token_Type> OPERATORS {
    {"(", Token_Type::OPEN_PARENTHESIS},
    {")", Token_Type::CLOSE_PARENTHESIS},
    {"+", Token_Type::ADDITION},
    {"-", Token_Type::SUBTRACTION},
    {"*", Token_Type::MULTIPLICATION},
    {"/", Token_Type::DIVISION}
  };

  /*
    Starting at index, for each subsequent digit add it to a single token of type NUMBER.
    Increments i to the last digit/character of the number.
  */
  Token consume_number (std::string_view input, size_t &i) {
    bool decimal_point_found {false};

    std::string value {};

    for (; i < input.size(); i++) {
      if (std::isdigit(input[i])) {
	value += input[i];
	// Spaces commas and apostrophes are all ignored
      } else if (std::isspace(input[i]) ||
		 input[i] == ',' ||
		 input[i] == '\'') {
	continue;
      } else if (input[i] == '.') {
	// Only one decimal point is allowed, the second one will result in an error.
	if (decimal_point_found) {
	  throw std::runtime_error {"Unexpected decimal point at index " + std::to_string(i) + " of input string \"" + std::string {input} + "\""};
	}

	decimal_point_found = true;
	value += '.';
      } else {
	// Something besides a decimal point, comma, apostrophe, digit, or whitespace, signalling something not part of this number.
	break;
      }
    }

    i--;
    return Token{Token_Type::NUMBER, value};
  }

  /*
    Convert a string and an array of possible operations to a vector of tokens.
  */
  std::vector<Token> tokenize (std::string_view input) {
    std::vector<Token> tokens {};

    // Start with something that isn't a number or a variable.
    Token_Type previous {Token_Type::ADDITION};

    for (size_t i {0}; i < input.size(); i++) {
      //    std::cout << i << ": " << input[i] << '\n';

      if (std::isspace(input[i])) continue;

      if (std::isdigit(input[i]) ||
	  // Numbers can also start with decimal points.
	  input[i] == '.') {
	// If the previous token was a variable, then insert a multiplication between them. x3 -> x * 3
	if (previous == Token_Type::VARIABLE) {
	  tokens.push_back( Token{OPERATORS.at("*"), "*"} );
	}

	tokens.push_back(consume_number(input, i));
	previous = Token_Type::NUMBER;
      } else if (std::isalpha(input[i])) {
	// If the previous token was a number, then insert a multiplication between them. 3x -> 3 * x
	if (previous == Token_Type::NUMBER) {
	  tokens.push_back( Token{OPERATORS.at("*"), "*"} );
	}

	tokens.push_back( Token{Token_Type::VARIABLE, input.substr(i, 1)} );
	previous = Token_Type::VARIABLE;
      } else {
	// Check if it is an operation.
	for (const auto &[op_str, op_token] : OPERATORS) {
	  if (input.substr(i).starts_with(op_str)) {
	    tokens.push_back( Token{op_token, std::string(op_str)} );

	    // If there was an operation, skip to the last character of the operation, then the for loop will increment to the next character.
	    i += op_str.size() - 1;
	  
	    // Don't keep searching for new operations.
	    break;
	  }
	}
      }
    }

    return tokens;
  }

  namespace syntax {
    
    //    void make_ast (const std::vector<Token> &tokens) {
      
    //    }

  } // namespace syntax
  
} // namespace lexer
