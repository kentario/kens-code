#include <iostream>
#include <memory>

#pragma once

#include "function.hpp"

enum class Token_Type {
  OPERATOR,
  VARIABLE,
  NUMBER
};

class Token {
private:
  Token_Type type;
  std::string value;
  size_t position;
public:
  Token (const Token_Type type, const size_t position) :
    type {type}, value {}, position {position} {}

  Token (const Token_Type type, const std::string &value, const size_t position) :
    type {type}, value {value}, position {position} {}

  Token_Type get_type () const {return type;}
  std::string get_value () const {return value;}
  size_t get_position () const {return position;}

  void push_letter (const char letter) {value.push_back(letter);}
};

std::ostream& operator<< (std::ostream &os, const Token &token) {
  os << "Type: ";
  switch (token.get_type()) {
  case Token_Type::OPERATOR:
    os << "operator ";
    break;
  case Token_Type::VARIABLE:
    os << "variable ";
    break;
  case Token_Type::NUMBER:
    os << "number ";
    break;
  }

  os << "Value: " << token.get_value() << " ";
  os << "Position: " << token.get_position();

  return os;
}

// Consume a variable from the input string starting from current_index. Increments current_index to be at the end of the variable.
Token consume_variable (const std::string &input, size_t &current_index) {
  Token variable {Token_Type::VARIABLE, current_index};

  while (std::isalpha(input[current_index])) {
    variable.push_letter(input[current_index]);
    current_index++;
  }

  // In case there is a whitespace character after the end of the variable, or this is the end of the string, go back a character to the end of the variable, then go forwards a character.
  current_index--;
  return variable;
}

// Consume a number from the input string starting from current_index. Increments current_index to be at the end of the number.
Token consume_number (const std::string &input, size_t &current_index) {
  Token number {Token_Type::NUMBER, current_index};

  bool decimal_found {false};

  while (std::isdigit(input[current_index]) || (input[current_index] == '.')) {
    if (input[current_index] == '.') {
      // Only allow one decimal point per number.
      if (decimal_found) {
	throw std::runtime_error {"Invalid number format: more than one decimal point found at postion " + std::to_string(current_index)};
      } else {
	decimal_found = true;
      }
    }
    number.push_letter(input[current_index]);
    current_index++;
  }

  current_index--;
  return number;
}

std::vector<Token> tokenize (const std::string &input, const std::vector<std::string> &operations) {
  std::vector<Token> tokens;

  for (size_t i {0}; i < input.size(); i++) {
    if (std::isspace(input[i])) continue;

    // Variables can only contain [a-zA-Z]+
    if (std::isalpha(input[i])) {
      tokens.push_back(consume_variable(input, i));
      continue;
    }

    // A number is a series of digits, with 0 or 1 decimals.
    if (std::isdigit(input[i]) || input[i] == '.') {
      tokens.push_back(consume_number(input, i));
      continue;
    }

    bool operation_found {false};
    for (const auto &operation : operations) {
      // Check if the operation would fit in the rest of the input.
      if (operation.size() + i > input.size()) continue;

      // Check if the strings match.
      if (operation == input.substr(i, operation.size())) {
	tokens.push_back(Token {Token_Type::OPERATOR, operation, i});
	i += operation.size() - 1;

	operation_found = true;
	continue;
      }
    }

    if (operation_found) continue;

    // If the character is not viable for a variable, a letter, or an operation, then throw an error.
    throw std::runtime_error {"Invalid character at position " + std::to_string(i) + ": '" + input[i] + "'"};
  }

  return tokens;
}

function_pointer token_to_function (const Token &token) {
  function_pointer output;
  switch (token.get_type()) {
  case Token_Type::VARIABLE:
    output = make_variable(token.get_value());
    break;
  case Token_Type::NUMBER:
    output = make_constant(std::stod(token.get_value()));
    break;
  case Token_Type::OPERATOR:
    if (token.get_value() == "+") {
      output = make_function<Addition>();
    } else if (token.get_value() == "*") {
      output = make_function<Multiplication>();
    } else if (token.get_value() == "-") {
      output = make_function<Subtraction>();
    } else if (token.get_value() == "/") {
      output = make_function<Division>();
    } else if (token.get_value() == "^") {
      output = make_function<Exponent>();
    } else if (token.get_value() == "(") {
      output = make_function<Open_Parenthesis>();
    } else if (token.get_value() == ")") {
      output = make_function<Close_Parenthesis>();
    } else {
      throw std::invalid_argument {"Unknown operator: '" + token.get_value() + "'"};
    }
  }

  return output;
}
