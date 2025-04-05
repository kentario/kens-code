#pragma once

#include <unordered_map>
#include <memory>
#include <any>
#include <type_traits>
#include <concepts>
#include <stdexcept>
#include <utility>
#include <cmath>
#include <string_view>

#include "token.hpp"

/* NAME: name of the operation and class
   SYMBOL: symbol that represents the operation
   IS_PREFIX: true or false value of whether the operation comes before or after the aragument
   OPERATION: brace enclosed block of code with a return statement. Evaluates the result of using this operation on input
 */
#define MAKE_UNARY_OPERATOR(NAME, SYMBOL, IS_PREFIX, OPERATION)            \
  template <Arithmetic N>                                                  \
  class NAME : public Unary_Operator<N> {                                  \
  public:                                                                  \
  /* Allows the derived class to use the constructor of the base class. */ \
    using Unary_Operator<N>::Unary_Operator;                               \
                                                                           \
    std::string_view get_symbol () const override {                        \
      return std::string_view {#SYMBOL};                                   \
    }                                                                      \
                                                                           \
    bool is_prefix () const override {return IS_PREFIX;}                   \
                                                                           \
    N operation (const N &input) const override {                          \
      OPERATION;                                                           \
    }                                                                      \
  };

/* NAME: name of the operation and class
   SYMBOL: symbol that represents the operation
   OPERATION: brace enclosed block of code with a return statement. Evaluated the result of using this operation on a and b.
 */
#define MAKE_BINARY_OPERATOR(NAME, SYMBOL, OPERATION)                                     \
  template <Arithmetic L, Arithmetic R>                                                   \
  class NAME : public Binary_Operator<L, R> {                                             \
  public:                                                                                 \
  /* Allows the derived class to use the constructor of the base class. */                \
    using Binary_Operator<L, R>::Binary_Operator;                                         \
                                                                                          \
    std::string_view get_symbol () const override {                                       \
      return std::string_view {#SYMBOL};                                                  \
    }                                                                                     \
                                                                                          \
    typename Binary_Operator<L, R>::N operation (const L &a, const R &b) const override { \
      OPERATION;                                                                          \
    }                                                                                     \
  };

namespace expression {

  template <typename T>
  concept Arithmetic = requires (T a) {
    {a + a} -> std::convertible_to<T>;
    {a - a} -> std::convertible_to<T>;
    {a * a} -> std::convertible_to<T>;
    {a / a} -> std::convertible_to<T>;

    a += a;
    a -= a;
    a *= a;
    a /= a;
  };

  // Forward declarations.
  template <Arithmetic N> class Expression;

  // Helps with functions that take Expression Pointers as input.
  /*
    The type T must:
    - Have a member alias `OUTPUT_TYPE` that represents the type it evaluates to.
    - Be a subclass of `Expression<OUTPUT_TYPE>`, i.e., `T` must inherit from `Expression<OUTPUT_TYPE>`.
  */
  template <typename T>
  concept Derived_From_Expression = requires (T a) {
    typename T::OUTPUT_TYPE;
    std::is_base_of_v<Expression<typename T::OUTPUT_TYPE>, T>;
  };

  /*
    Expressions will optionally take a map of variable names to variable values as input.
    For example, my_function->evaluate({{'x', 3}, {'a', 34}, {'y', 1}})

    To access a value, the value must be converted from std::any to the desired type.
    In Variable, it is used return std::any_cast<N>(values.at(name)) to go from std::any to the type of the variable;
  */
  template <Arithmetic N>
  using var_values = std::unordered_map<std::string, std::any>;

  // Makes using polymorphism slightly easier to read.
  template <Arithmetic N>
  using Expression_Pointer = std::unique_ptr<Expression<N>>;

  // Base abstract class for all functions.
  template <Arithmetic N>
  class Expression {
  public:
    using OUTPUT_TYPE = N;
    
    virtual N evaluate (const var_values<N> &values = {}) const = 0;

    virtual std::string_view get_symbol () const = 0;

    // Very usefull stuff about inheritance, virtual and friend functions, and overloading https://www.learncpp.com/cpp-tutorial/printing-inherited-classes-using-operator/
    // This is using the section titled "A more flexible solution."
    virtual std::ostream& print (std::ostream &os) const = 0;
  };

  template <typename N>
  std::ostream& operator<<(std::ostream &os, const Expression<N>& expr) {
    return expr.print(os);
    
    return os;
  }

  namespace operators {

    // Abstract class for all unary operations.
    template <Arithmetic N>
    class Unary_Operator : public Expression<N> {
    protected:
      Expression_Pointer<N> arg;
    public:
      /*
	When calling this constructor, the inputs either need to be rvalues, or std::move(lvalue).
	If std::move(lvalue) is used, the lvalue will be set to nullptr.
      */
      Unary_Operator (Expression_Pointer<N> &&arg) :
	arg {std::move(arg)} {}
      
      N evaluate (const var_values<N> &values = {}) const override {
	return operation(arg->evaluate(values));
      }

      virtual bool is_prefix () const = 0;

      virtual N operation (const N &input) const = 0;

      std::ostream& print (std::ostream &os) const override {
	os << '(';

	if (this->is_prefix()) {
	  // Using a pointer so that polymorphism applies, and there is a different symbol used for different derived classes.
	  os << this->get_symbol() << *arg;
	} else {
	  os << *arg << this->get_symbol();
	}
	
	os << ')';
	return os;
      }
    };

    MAKE_UNARY_OPERATOR(Negate, -, true, {return -input;})
    MAKE_UNARY_OPERATOR(Square_Root, sqrt, true, {return std::sqrt(input);})
    
    // Abstract class for all binary operations.
    template <Arithmetic L, Arithmetic R>
    /*
      From https://en.cppreference.com/w/cpp/types/common_type:
      "Determines the common type among all types T..., that is the type all T... can be implicitly converted to.
      If such a type exists (as determined according to the rules below), the member type names that type. Otherwise, there is no member type."
    */
    class Binary_Operator : public Expression<typename std::common_type<L, R>::type> {
    protected:
      Expression_Pointer<L> left;
      Expression_Pointer<R> right;
    public:
      using N = std::common_type<L, R>::type;
      
      /*
	When calling this constructor, the inputs either need to be rvalues, or std::move(lvalue).
      */
      Binary_Operator (Expression_Pointer<L> &&left, Expression_Pointer<R> &&right) :
	left {std::move(left)}, right {std::move(right)} {}

      N evaluate (const var_values<N> &values = {}) const override {
	return operation(left->evaluate(values), right->evaluate(values));
      }

      virtual N operation (const L &a, const R &b) const = 0;

      std::ostream& print (std::ostream &os) const override {
	os << '(' << *left << this->get_symbol() << *right << ')';

	return os;
      }
    };

    MAKE_BINARY_OPERATOR(Addition, +, {return a + b;})
    MAKE_BINARY_OPERATOR(Subtraction, -, {return a - b;})
    MAKE_BINARY_OPERATOR(Multiplication, *, {return a * b;})
    MAKE_BINARY_OPERATOR(Division, /, {return a / b;})

  } // namespace operators

  namespace operands {
    
    template <Arithmetic N>
    class Variable : public Expression<N> {
    private:
      const std::string name;
    public:
      Variable (const std::string &name) :
	name {name} {}

      Variable (const char name) :
	name(1, name) {}

      std::string_view get_symbol () const override {
	return name;
      }

      N evaluate (const var_values<N> &values = {}) const override {
	if (!values.contains(name)) {
	  throw std::runtime_error {"Value for variable '" + std::string {name} + "' not found in input values"};
	}

	return std::any_cast<N>(values.at(name));
      }

      std::ostream& print (std::ostream &os) const override {
	os << name;
	
	return os;
      }
    };

    template <Arithmetic N>
    class Number : public Expression<N> {
    private:
      const N value;
    public:
      Number (const N &value) :
	value {value} {}

      std::string_view get_symbol () const override {
	return std::string_view {"number"};
      }

      N evaluate (const var_values<N> &values = {}) const override {
	(void) values;
	return value;
      }

      std::ostream& print (std::ostream &os) const override {
	os << value;

	return os;
      }
    };
    
  } // namespace operands
  
} // namespace math_expressions
