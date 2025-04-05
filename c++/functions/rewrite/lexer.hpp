#pragma once

#include <unordered_map>
#include <string_view>
#include <vector>
#include <utility>
#include <stdexcept>

#include "token.hpp"

namespace lexer {

  using namespace token;

  const std::unordered_map<std::string_view, Token_Type> OPERATORS {
    {"(", Token_Type::OPEN_PARENTHESIS},
    {")", Token_Type::CLOSE_PARENTHESIS},
    {"+", Token_Type::ADDITION},
    {"-", Token_Type::SUBTRACTION},
    {"*", Token_Type::MULTIPLICATION},
    {"/", Token_Type::DIVISION},
    {"sqrt", Token_Type::SQUARE_ROOT}
  };

  /*
    Starting at index i, for each subsequent digit add it to a single token of type NUMBER.
    After returning, i will be at the last digit/character of the number.
  */
  Token consume_number (std::string_view input, size_t &i) {
    bool decimal_point_found {false};

    // The position of the first digit of the number.
    const size_t start {i};

    for (; i < input.size(); i++) {
      if (std::isdigit(input[i])) {
	// Spaces, commas, and apostrophes are all ignored.
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
      } else {
	// Something besides a decimal point, comma, apostrophe, digit, or whitespace, signalling something not part of this number.
	break;
      }
    }

    /*
      Example for finding the size of a number:
      0 1 2 3 4 5 6 7 8
      a b g 9 8 3 d e b
            ^     ^        6 - 3 = 3
     */
    const size_t size {i-- - start};
    // Decrease i because the loop in the tokenize function will increment i after this functino returns.

    // The length of the number is 
    return Token {Token_Type::NUMBER, input.substr(start, size)};
  }

  /*
    Starting at index i, will return a token of the variable at that position.
    After returning, i will be at the last character of the variable name.
   */
  Token consume_variable (std::string_view input, size_t &i) {
    /*
       A variable is any single letter, optionally followed by _
       If a character is followed by _, it, along with any more characters, numbers, or _ after it are included within the variable name.
     */
    
    // Single letter variable. When the next letter doesn't exist, or neither the next nor current letter are underscores.
    if (i + 1 >= input.size() || (input[i + 1] != '_' && input[i] != '_')) return Token {Token_Type::VARIABLE, input.substr(i, 1)};
    
    const size_t start {i};

    while (i < input.size() && (std::isalnum(input[i]) || input[i] == '_')) {
      i++;
    }

    const size_t size {i-- - start};

    return Token {Token_Type::VARIABLE, input.substr(start, size)};
  }
  
  /*
    Convert a string to a vector of tokens.
  */
  // TODO: Currenty any unrecognized characters will be ignored, instead make an error happen, but have the tokenizing continue so as to catch as many errors as possible.
  std::vector<Token> tokenize (std::string_view input) {
    std::vector<Token> tokens {};

    // i is always pointing at the current character being evaluated.
    for (size_t i {0}; i < input.size(); i++) {
      if (std::isspace(input[i])) continue;

      /*
	== NUMBERS ==
       */
      if (std::isdigit(input[i]) ||
	  // Numbers can also start with decimal points.
	  input[i] == '.') {

	tokens.push_back(consume_number(input, i));

	/*
	  == OPERATORS ==
	 */
      } else {
	bool op_found {false};
	// Check if the current position is the start of an operation.
	for (const auto &[op_str, op_token] : OPERATORS) {
	  if (input.substr(i).starts_with(op_str)) {
	    tokens.push_back( Token {op_token, op_str} );

	    // If there was an operation, skip to the last character of the operation.
	    i += op_str.size() - 1;
	    // -1 because the for loop will increment to the next character after it.
	  
	    // Don't keep searching for new operations.
	    op_found = true;
	    break;
	  }
	}
	if (op_found) continue;
	
	// If it wasn't an operation, then check if it was a variable:
	/*
	  == VARIABLES ==
	*/
	if (std::isalpha(input[i]) ||
	    // Variables can start with '_'
	    input[i] == '_') {
	  tokens.push_back( consume_variable(input, i));
	} else {
	  // If it is something unknown:
	  // 	  throw std::runtime_error {"Unexpected decimal point at index " + std::to_string(i) + " of input string \"" + std::string {input} + "\""};
	  throw std::runtime_error {"Unexpected character at index " + std::to_string(i) + ": '" + std::string(1, input[i]) + "'"};
	}
      }
    }

    tokens.push_back( Token {Token_Type::END_OF_FILE} );
    return tokens;
  }

} // namespace lexer
