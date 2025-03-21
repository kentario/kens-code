#include <iostream>
#include <array>
#include <string>
#include <memory>
#include <unordered_map>
#include <string_view>

#include "expression-factory.hpp"
#include "token.hpp"
#include "parser.hpp"

using namespace math_expressions;
using namespace factory;
using namespace operands;
using namespace operators;

int main (/* int argc, char *argv[] */) {
  std::string_view input {"(124+ 2)34.432/.2a + a 3,32"};

  std::cout << "input: " << input << "\n\n";

  auto tokens = lexer::tokenize(input);

  /*  for (auto token : tokens) {
    std::cout << token << '\n';
    }*/

  // namespace stuff not needed, just in these examples incase it is too confusing to search through the code for what namespaces shoul be used.
  auto var_x = std::make_unique<operands::Variable<double>>('x');
  auto fifteen = std::make_unique<operands::Number<int>>(15);
  auto var_plus_fifteen = factory::make_binary_operator<operators::Addition>
    (
     std::move(var_x), std::move(fifteen)
     );
  std::cout << var_plus_fifteen->evaluate({{'x', 3.0}}) << '\n';
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

  /*
               	   (-b + sqrt(b^2 - 4 * a * c))
    q_p(a, b, c) = ----------------------------
                             (2 * a)

               	   (-b - sqrt(b^2 - 4 * a * c))
    q_m(a, b, c) = ----------------------------
                             (2 * a)
  */

  auto q_p = make_binary_operator<operators::Addition>
    (
     std::move(left),
     std::move(right)
     );

    std::cout << q_p->evaluate({{'a', 2.8}, {'b', 3.1}, {'c', -0.4}}) << '\n';

    std::cout << *q_p << '\n';

  
  return 0;
}
