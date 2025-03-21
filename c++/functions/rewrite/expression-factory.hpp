#pragma once

#include <memory>

#include "expression.hpp"

namespace math_expressions {

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
    auto make_unary_operator(std::unique_ptr<T>&& arg) {
      return std::make_unique<
	OPERATOR_TYPE<
	  typename T::OUTPUT_TYPE
	  >>(std::move(arg));
    }
    
  } // namespace factory
  
} // namespace math_expressions
