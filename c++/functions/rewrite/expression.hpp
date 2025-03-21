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

namespace math_expressions {

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
  using var_values = std::unordered_map<char, std::any>;

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

      virtual N operation (const N &input) const = 0;

      virtual bool is_prefix () const = 0;

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

    template <Arithmetic N>
    class Negate : public Unary_Operator<N> {
    public:
      using Unary_Operator<N>::Unary_Operator;

      std::string_view get_symbol () const override {
	return std::string_view {"-"};
      }
      
      N operation (const N &input) const override {
	return -input;
      }

      bool is_prefix () const override {return true;}
    };

    template <Arithmetic N>
    class Square_Root : public Unary_Operator<N> {
    public:
      using Unary_Operator<N>::Unary_Operator;

      std::string_view get_symbol () const override {
	return std::string_view {"sqrt"};
      }

      N operation (const N &input) const override {
	return std::sqrt(input);
      }

      bool is_prefix () const override {return true;}
    };
    
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
    
    template <Arithmetic L, Arithmetic R>
    class Addition : public Binary_Operator<L, R> {
    public:
      // Allows the derived class to use the constructor of the base class.
      using Binary_Operator<L, R>::Binary_Operator;

      std::string_view get_symbol () const override {
	return std::string_view {"+"};
      }
      
      typename Binary_Operator<L, R>::N operation (const L &a, const R &b) const override {
	return a + b;
      }
    };

    template <Arithmetic L, Arithmetic R>
    class Subtraction : public Binary_Operator<L, R> {
    public:
      // Allows the derived class to use the constructor of the base class.
      using Binary_Operator<L, R>::Binary_Operator;

      std::string_view get_symbol () const override {
	return std::string_view {"-"};
      }
      
      typename Binary_Operator<L, R>::N operation (const L &a, const R &b) const override {
	return a - b;
      }
    };

    template <Arithmetic L, Arithmetic R>
    class Multiplication : public Binary_Operator<L, R> {
    public:
      // Allows the derived class to use the constructor of the base class.
      using Binary_Operator<L, R>::Binary_Operator;

      std::string_view get_symbol () const override {
	return std::string_view {"*"};
      }
      
      typename Binary_Operator<L, R>::N operation (const L &a, const R &b) const override {
	return a * b;
      }
    };

    template <Arithmetic L, Arithmetic R>
    class Division : public Binary_Operator<L, R> {
    public:
      // Allows the derived class to use the constructor of the base class.
      using Binary_Operator<L, R>::Binary_Operator;

      std::string_view get_symbol () const override {
	return std::string_view {"/"};
      }
      
      typename Binary_Operator<L, R>::N operation (const L &a, const R &b) const override {
	return a / b;
      }
    };
    
  } // namespace operators

  namespace operands {
    
    template <Arithmetic N>
    class Variable : public Expression<N> {
    private:
      const char name;
    public:
      Variable (const char name) :
	name {name} {}

      std::string_view get_symbol () const override {
	return std::string_view {&name, 1};
      }

      N evaluate (const var_values<N> &values = {}) const override {
	if (!values.contains(name)) {
	  throw std::runtime_error {"Variable '" + std::string {name} + "' not found in input values"};
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
	//	return std::string_view {std::to_string(value)};
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
