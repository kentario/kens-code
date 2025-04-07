#include <iostream>
#include <array>
#include <string>
#include <memory>
#include <unordered_map>
#include <string_view>

#include "token.hpp"
#include "lexer.hpp"
#include "parser.hpp"
#include "expression.hpp"
#include "expression-factory.hpp"

using namespace expression;
using namespace factory;
using namespace operands;
using namespace operators;

int main (int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Correct usage: " << argv[0] << " <expression>\n";
    return 1;
  }

  std::string_view input {argv[1]};

  std::cout << "input: " << input << "\n\n";

  auto tokens = lexer::tokenize(input);
  for (auto token : tokens) {
    std::cout << token << '\n';
  } std::cout << '\n';
  
  auto expr = parser::parse(tokens);
  std::cout << *expr << '\n';
  std::cout << "evaluate: " << expr->evaluate({{"a", -0.8}, {"b", 2.71299}, {"c", -2.3}}) << '\n';
  

  /*
  // namespace stuff not needed, just in these examples in case it is too confusing to search through the code for what namespaces should be used.
  auto var_x = std::make_unique<expression::operands::Variable<double>>('x');
  auto fifteen = std::make_unique<expression::operands::Number<int>>(15);
  auto var_plus_fifteen = factory::make_binary_operator<expression::operators::Addition>
    (
     std::move(var_x), std::move(fifteen)
     );
  std::cout << var_plus_fifteen->evaluate({{"x", 3.0}}) << '\n';
  std::cout << *var_plus_fifteen << '\n';


  // -b/(2a)
  auto left = make_binary_operator<Division>
    (
     make_unary_operator<Negate>
     (
      std::make_unique<Variable<double>>('b')					     
      ),
     make_binary_operator<Multiplication>
     (
      std::make_unique<Number<int>>(2),
      std::make_unique<Variable<double>>('a')
      )
     );

  // sqrt(bb - 4ac)/(2a)
  auto right = make_binary_operator<operators::Division>
    (
     make_unary_operator<operators::Square_Root>
     (
      make_binary_operator<operators::Subtraction>
      (
       make_binary_operator<operators::Multiplication>
       (
	std::make_unique<Variable<double>>('b'),
	std::make_unique<Variable<double>>('b')
	),
       make_binary_operator<operators::Multiplication>
       (
	make_binary_operator<operators::Multiplication>
	(
	 std::make_unique<Number<int>>(4),
	 std::make_unique<Variable<double>>('a')
	 ),
	std::make_unique<Variable<double>>('c')
	)
       )
      ),
     make_binary_operator<operators::Multiplication>
     (
      std::make_unique<Number<int>>(2),
      std::make_unique<Variable<double>>('a')
      )
     );
  */
  /*
               	   (-b + sqrt(b^2 - 4 * a * c))
    q_p(a, b, c) = ----------------------------
                             (2 * a)

               	   (-b - sqrt(b^2 - 4 * a * c))
    q_m(a, b, c) = ----------------------------
                             (2 * a)
  */
  /*
  auto q_p = make_binary_operator<operators::Addition>
    (
     std::move(left),
     std::move(right)
     );

    std::cout << q_p->evaluate({{"a", 2.8}, {"b", 3.1}, {"c", -0.4}}) << '\n';

    std::cout << *q_p << '\n';
  */

  return 0;
}
