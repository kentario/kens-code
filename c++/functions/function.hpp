#include <iostream>
#include <cmath>
#include <string>
#include <memory>
#include <stdexcept>
#include <vector>

#pragma once

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

  bool node_at_exists (size_t index) {
    if (index == 1) return next.get();
    if (next) return next->node_at_exists(index - 1);
    return false;
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
