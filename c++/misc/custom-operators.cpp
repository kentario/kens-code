#include <iostream>

// To use this, enter the values for the types and name.
// In the code, make sure to have a return statement, and curly braces are recommended.
// To refer to the values on the left and right of the expression, refer to them as left and right.
// To use the custom operator, either do left *NAME* right.
// or have a #define NAME *NAME* and then do left NAME right.
#define MCBO(RETURN_TYPE, LEFT_TYPE, NAME, RIGHT_TYPE, CODE) MAKE_CUSTOM_BINARY_OPERATOR(RETURN_TYPE, LEFT_TYPE, NAME, RIGHT_TYPE, CODE)
#define MAKE_CUSTOM_BINARY_OPERATOR(RETURN_TYPE, LEFT_TYPE, NAME, RIGHT_TYPE, CODE) \
  struct custom_##NAME##_operator {					\
  } NAME;								\
  struct custom_##NAME##_proxy {					\
    const LEFT_TYPE left;						\
    custom_##NAME##_proxy (const LEFT_TYPE &left) : left{left} {}	\
    operator LEFT_TYPE() const {					\
      return left;							\
    }									\
  };									\
  custom_##NAME##_proxy operator* (LEFT_TYPE left, custom_##NAME##_operator custom_operator) { \
    return custom_##NAME##_proxy {left};				\
  }									\
  RIGHT_TYPE operator* (custom_##NAME##_proxy left, RIGHT_TYPE right) {	\
    CODE								\
      }


// Use value to refer to the value being operated on.
// Curly braces are highly recommended.
#define MCO_POSTFIX(RETURN_TYPE, INPUT_TYPE, NAME, CODE) MAKE_CUSTOM_OPERATOR_POSTFIX(RETURN_TYPE, INPUT_TYPE, NAME, CODE)
#define MAKE_CUSTOM_OPERATOR_POSTFIX(RETURN_TYPE, INPUT_TYPE, NAME, CODE) \
  struct custom_##NAME##_operator {					\
  } NAME;								\
  RETURN_TYPE operator* (const INPUT_TYPE &value, custom_##NAME##_operator custom_operator) { \
    CODE\
      }

#define MCO_PREFIX(RETURN_TYPE, INPUT_TYPE, NAME, CODE) MAKE_CUSTOM_OPERATOR_PREFIX(RETURN_TYPE, INPUT_TYPE, NAME, CODE)
#define MAKE_CUSTOM_OPERATOR_PREFIX(RETURN_TYPE, INPUT_TYPE, NAME, CODE) \
  struct custom_##NAME##_operator {					\
  } NAME;								\
  RETURN_TYPE operator* (custom_##NAME##_operator custom_operator, const INPUT_TYPE &value) {	\
    CODE\
      }


MCBO(int, int, plus, int, {
    return left + right;
  })
#define plus *plus*

MCBO(int, int, to_the_power_of, int, {
    int product {1};
    for (int i {0}; i < right; i++) {
      product *= left;
    }
    return product;
  })
#define to_the_power_of *to_the_power_of*

MCO_POSTFIX(int, int, factorial, {
    int product {1};
    for (int i {value}; i > 0; i--) {
      product *= i;
    }
    return product;
  })
#define factorial *factorial

MCO_PREFIX(int, int, name, {
    return value * 2;
  })
#define name name*

int main () {
  std::cout << 3 plus 5 << "\n";
  std::cout << 2 to_the_power_of 5 << "\n";
  std::cout << 5 factorial << "\n";
  std::cout << name 13 << "\n";
  
  return 0;
}
