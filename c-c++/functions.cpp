#include <iostream>
#include <cmath>
#include <string>
#include <memory>
#include <stdexcept>
#include <vector>

enum class Function_Type {
  ADDITION,
  MULTIPLICATION,
  SUBTRACTION,
  DIVISION,
  EXPONENT,
  VARIABLE,
  CONSTANT,
  OPEN_PARENTHESIS,
  CLOSE_PARENTHESIS
};

class Function {
protected:
  // Implementation of a linked list.
  std::shared_ptr<Function> next;

  // A function is an operation on two other functions.
  std::shared_ptr<Function> a;
  std::shared_ptr<Function> b;

  char symbol;
  Function_Type type;
public:
  Function (const Function_Type type) : type {type} {}

  Function (const std::shared_ptr<Function> &next, const Function_Type type) :
    next {next}, a {nullptr}, b {nullptr}, symbol {}, type {type} {}

  Function (const std::shared_ptr<Function> &next, const char symbol, const Function_Type type) :
    next {next}, a {nullptr}, b {nullptr}, symbol {symbol}, type {type} {}

  Function (const std::shared_ptr<Function> &a, const std::shared_ptr<Function> &b, const char symbol, const Function_Type type) :
    next {nullptr}, a {a}, b {b}, symbol {symbol}, type {type} {}

  /*
  std::shared_ptr<Function>& operator[] (size_t index) {
    if (index == 1) return next;
    if (next) return (*next)[index - 1];
    throw std::out_of_range {"Index out of bounds"};
  }
  */

  std::shared_ptr<Function>& node_at (size_t index) {
    if (index == 1) return next;
    if (next) return next->node_at(index - 1);
    throw std::out_of_range {"Index out of bounds"};
  }

  char get_symbol () const {return symbol;}

  Function_Type get_type () const {return type;}

  std::shared_ptr<Function> get_next() const {return next;}
  std::shared_ptr<Function> get_a () const {return a;}
  std::shared_ptr<Function> get_b () const {return b;}

  void set_next (std::shared_ptr<Function> new_next) {next = new_next;}
  void set_a (std::shared_ptr<Function> new_a) {a = new_a;}
  void set_b (std::shared_ptr<Function> new_b) {b = new_b;}

  virtual double evaluate (double input) const = 0;

  virtual std::ostream& print (std::ostream &os) const {
    os << "(";
    if (a) a->print(os);
    os << " " << symbol << " ";
    if (b) b->print(os);
    os << ")";

    return os;
  }
};

std::ostream& operator<< (std::ostream &os, const Function &func) {
  return func.print(os);
}

class Addition : public Function {
public:
  Addition (const std::shared_ptr<Function> next) : Function {next, '+', Function_Type::ADDITION} {}
  Addition (const std::shared_ptr<Function> a, const std::shared_ptr<Function> b) : Function {a,  b, '+', Function_Type::ADDITION} {}

  double evaluate (double input) const override {
    return a->evaluate(input) + b->evaluate(input);
  }
};

class Multiplication : public Function {
public:
  Multiplication (const std::shared_ptr<Function> next) : Function {next, '*', Function_Type::MULTIPLICATION} {}
  Multiplication (const std::shared_ptr<Function> a, const std::shared_ptr<Function> b) : Function {a,  b, '*', Function_Type::MULTIPLICATION} {}

  double evaluate (double input) const override {
    return a->evaluate(input) * b->evaluate(input);
  }
};

class Subtraction : public Function {
public:
  Subtraction (const std::shared_ptr<Function> next) : Function {next, '-', Function_Type::SUBTRACTION} {}
  Subtraction (const std::shared_ptr<Function> a, const std::shared_ptr<Function> b) : Function {a,  b, '-', Function_Type::SUBTRACTION} {}

  double evaluate (double input) const override {
    return a->evaluate(input) - b->evaluate(input);
  }
};

class Division : public Function {
public:
  Division (const std::shared_ptr<Function> next) : Function {next, '/', Function_Type::DIVISION} {}
  Division (const std::shared_ptr<Function> a, const std::shared_ptr<Function> b) : Function {a,  b, '/', Function_Type::DIVISION} {}

  double evaluate (double input) const override {
    double denominator {b->evaluate(input)};
    if (denominator != 0) {
      return a->evaluate(input)/b->evaluate(input);
    } else {
      throw std::runtime_error {"Division by zero"};
    }
  }
};

class Exponent : public Function {
public:
  Exponent (const std::shared_ptr<Function> next) : Function {next, '^', Function_Type::EXPONENT} {}
  Exponent (const std::shared_ptr<Function> a, const std::shared_ptr<Function> b) : Function {a, b, '^', Function_Type::EXPONENT} {}

  double evaluate (double input) const override {
    return std::pow(a->evaluate(input), b->evaluate(input));
  }
};

class Constant : public Function {
private:
  const double value;
public:
  explicit Constant (const double value) : Function {Function_Type::CONSTANT}, value {value} {}
  Constant (const std::shared_ptr<Function> next, const double value) : Function {next, Function_Type::CONSTANT}, value {value} {}

  double evaluate (double input) const override {return value;}
  double evaluate () const {return value;}
  operator double() const {return value;}

  std::ostream& print (std::ostream &os) const override {
    os << value;
    return os;
  }
};

class Variable : public Function {
private:
  const std::string name;
public:
  explicit Variable (const std::string &name) : Function {Function_Type::VARIABLE}, name {name} {}
  Variable (const std::shared_ptr<Function> next, const std::string &name) : Function {next, Function_Type::VARIABLE}, name {name} {}

  std::string get_name () const {return name;}

  double evaluate (double input) const override {return input;}

  std::ostream& print (std::ostream &os) const override {
    os << name;
    return os;
  }
};

class Open_Parenthesis : public Function {
public:
  explicit Open_Parenthesis (const std::shared_ptr<Function> next)  : Function {next, Function_Type::OPEN_PARENTHESIS} {}

  double evaluate (double input) const override {throw std::logic_error {"Evaluation of open parenthesis not allowed"};}

  std::ostream& print (std::ostream &os) const override {
    os << "(";
    return os;
  }
};

class Close_Parenthesis : public Function {
public:
  explicit Close_Parenthesis (const std::shared_ptr<Function> next)  : Function {next, Function_Type::CLOSE_PARENTHESIS} {}

  double evaluate (double input) const override {throw std::logic_error {"Evaluation of close parenthesis not allowed"};}

  std::ostream& print (std::ostream &os) const override {
    os << ")";
    return os;
  }
};

std::shared_ptr<Function> make_constant (const double value) {
  return std::make_shared<Constant>(value);
}

std::shared_ptr<Function> make_variable (const std::string &name) {
  return std::make_shared<Variable>(name);
}

template <typename T>
std::shared_ptr<Function> make_function (const std::shared_ptr<Function> a, const std::shared_ptr<Function> b) {
  return std::make_shared<T>(a, b);
}

template <typename T>
std::shared_ptr<Function> make_function (const std::shared_ptr<Function> next = nullptr) {
  return std::make_shared<T>(next);
}

bool is_operator (const std::shared_ptr<const Function> &function) {
  return (function->get_type() == Function_Type::ADDITION ||
	  function->get_type() == Function_Type::MULTIPLICATION ||
	  function->get_type() == Function_Type::DIVISION ||
	  function->get_type() == Function_Type::SUBTRACTION ||
	  function->get_type() == Function_Type::EXPONENT);
}

bool is_operand (const std::shared_ptr<const Function> &function) {
  return (is_operator(function) ||
	  function->get_type() == Function_Type::CONSTANT ||
	  function->get_type() == Function_Type::VARIABLE);
}

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

std::shared_ptr<Function> token_to_function (const Token &token) {
  std::shared_ptr<Function> output;
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

/*
beginning = x
beginning->get_next = ^
beginning->get_next->get_next = 2
beginning->get_next->get_next->get_next = +
beginning->get_next->get_next->get_next->get_next = 1
beginning->get_next->get_next->get_next->get_next->get_next = NULL
end = NULL
*/

// Recursively generates a tree from the linked list of a function, going from [beginning, end).
std::shared_ptr<Function> generate_tree (const std::shared_ptr<Function> &beginning, const std::shared_ptr<Function> &end) {
  // Print out the linked list.
  for (auto p = beginning; p != end; p = p->get_next()) {
    std::cout << *p << " ";
  }
  std::cout << "\n";

  if (beginning->get_next() == end) return beginning;

  // Check if it is a simple form with just addition and subtraction such as 3 + 2 - 1 - 4 + 2
  // Or with just multiplication and division.
  // A simple form should have an odd number of elements.
  // Even indexes should be operands, and odd indexes should be operators.
  bool is_simple {true};
  size_t i {0};
  Function_Type types[2];
  for (auto p = beginning; p != end; p = p->get_next(), i++) {
    if (i % 2 == 0) {
      // Should be an operand.
      if (!is_operand(p)) {
	is_simple = false;
	break;
      }
    } else {
      // Should be an operator.
      if (i == 1) {
	// If it is the first operator, then find out if it is addition and subtraction, or multiplication and division.
	switch (p->get_type()) {
	case Function_Type::ADDITION:
	case Function_Type::SUBTRACTION:
	  types[0] = Function_Type::ADDITION;
	  types[1] = Function_Type::SUBTRACTION;
	  break;
	case Function_Type::MULTIPLICATION:
	case Function_Type::DIVISION:
	  types[0] = Function_Type::MULTIPLICATION;
	  types[1] = Function_Type::DIVISION;
	}
      } else {
	// If it is not the first operator, then check if it is in the same category as the first operator.
	if (p->get_type() != types[0] &&
	    p->get_type() != types[1]) {
	  is_simple = false;
	  break;
	}
      }
    }
  }

  // If there are an even number of nodes, then it is not simple.
  if (i % 2 == 0) is_simple = false;

  std::cout << "is simple: " << is_simple << "\n";

  /*
	  +
	 / \
        -   2
       / \
      +   5
     / \
    3   x
   */
  if (is_simple) {
    auto root = beginning->get_next();
    auto p = beginning;
    for (; root->node_at(2); root->set_next(root->node_at(2)), p = root, root = root->get_next()) {
      root->set_a(p);
      root->set_b(p->node_at(2));
    }
    
    return p;
  }

  return nullptr;
}

std::shared_ptr<Function> build_function (const std::vector<Token> &tokens) {
  std::shared_ptr<Function> function {token_to_function(tokens[0])};

  // Converts the vector of tokens to a linked list of functions.
  for (size_t i {1}; i < tokens.size(); i++) {
    function->node_at(i) = token_to_function(tokens[i]);
  }

  function = generate_tree(function, nullptr);

  return function;
}

int main () {
  // Current list of possible operations:
  const std::vector<std::string> operations {"+", "*", "-", "/", "^", "(", ")"};

  //                     0123456789
  std::string function {"9 - 5 + 7 + x"};
  std::vector<Token> tokenized_function {tokenize(function, operations)};
  for (const auto &token : tokenized_function) {
    std::cout << token << "\n";
  }
  std::cout << "\n";

  auto g = build_function(tokenized_function);
  std::cout << "Formula of g: " << *g << "\n";

  /*auto f = make_function<Exponent>(make_function<Addition>(make_constant(2), make_constant(5.1)), make_variable("X"));

  std::cout << f->evaluate(3) << "\n";

  // Print the tree of f.
  std::cout << *f << "\n";*/

  return 0;
}
