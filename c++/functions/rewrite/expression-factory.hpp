#pragma once

#include <memory>

#include "expression.hpp"
#include "token.hpp"

namespace expression {

  namespace factory {
    
    /*
      This is a template template paramter.
      The first section, template <typename, typename> typename OPERATOR_TYPE,
      means that this template expects a template for its first argument.
      This template that it expects is the OPERATOR_TYPE of the binary operator, and should have 2 typename arguments, in this case the left and right argument types.

      std::unordered_map is a template (template <typename, typename>), int is a typename, std::unordered_map<char, int> is a typename
      operators::Addition is a template (template <typename, typename>).
    */
    /*
      Returns a unique_ptr to a Binary_Operator of the specified type.
    */
    template <template <typename, typename> typename OPERATOR_TYPE,
	      Derived_From_Expression L, Derived_From_Expression R>
    auto make_binary_operator (std::unique_ptr<L> &&left, std::unique_ptr<R> &&right) {
      return std::make_unique<
	OPERATOR_TYPE<
	  typename L::OUTPUT_TYPE, typename R::OUTPUT_TYPE
	  >>(std::move(left), std::move(right));
    }

    /*
      Returns a unique_ptr to a Unary_Operator of the specified type.
    */
    template <template <typename> typename OPERATOR_TYPE,
	      Derived_From_Expression T>
    auto make_unary_operator(std::unique_ptr<T>&& expr) {
      return std::make_unique<
	OPERATOR_TYPE<
	  typename T::OUTPUT_TYPE
	  >>(std::move(expr));
    }
    
    // Versions of these functions that take in a Token_Type instead of a template parameter.
    template <Derived_From_Expression L, Derived_From_Expression R>
    // Using auto for the return type doesn't work because there are multiple return types within this function.
    // To fix this, the return type is explicitly written out how it was written in the Binary_Operator class.
    Expression_Pointer<typename std::common_type<typename L::OUTPUT_TYPE, typename R::OUTPUT_TYPE>::type>
    make_binary_operator (const token::Token_Type type, std::unique_ptr<L> &&left, std::unique_ptr<R> &&right) {
      switch (type) {
      case token::Token_Type::ADDITION:
	return make_binary_operator<operators::Addition>(std::move(left), std::move(right));
      case token::Token_Type::SUBTRACTION:
	return make_binary_operator<operators::Subtraction>(std::move(left), std::move(right));
      case token::Token_Type::MULTIPLICATION:
	return make_binary_operator<operators::Multiplication>(std::move(left), std::move(right));
      case token::Token_Type::DIVISION:
	return make_binary_operator<operators::Division>(std::move(left), std::move(right));
      default:
	throw std::runtime_error("Unsupported binary operator");
      }
    }

    // Versions of these functions that take in a Token_Type instead of a template parameter.
    template <Derived_From_Expression T>
    Expression_Pointer<typename T::OUTPUT_TYPE>
    make_unary_operator (const token::Token_Type type, std::unique_ptr<T> &&expr) {
      switch (type) {
      case token::Token_Type::SUBTRACTION:
	return make_unary_operator<operators::Negate>(std::move(expr));
      case token::Token_Type::SQUARE_ROOT:
	return make_unary_operator<operators::Square_Root>(std::move(expr));
      default:
	throw std::runtime_error("Unsupported unary operator");
      }
    }

  } // namespace factory
  
} // namespace math_expressions
