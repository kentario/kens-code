#include <iostream>
#include <array>
#include <string>
#include <memory>
#include <unordered_map>

#include "expression-factory.hpp"
#include "token.hpp"
#include "parser.hpp"

using namespace math_expressions;

int main (int argc, char *argv[]) {
  std::string input {"(124+ 2)34.432/.2a + a 3,32"};

  std::cout << input << "\n\n";

  auto tokens = lexer::tokenize(input);
  /*
    for (auto token : tokens) {
    std::cout << token << '\n';
    }*/

  /*
               	   (-b + sqrt(b^2 - 4 * a * c))
    q_p(a, b, c) = ----------------------------
                             (2 * a)

               	   (-b - sqrt(b^2 - 4 * a * c))
    q_m(a, b, c) = ----------------------------
                             (2 * a)
  */


  
  return 0;
}
