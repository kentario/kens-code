#include <iostream>
#include <string>
#include <vector>

/*
 * FOLDING
 * 1) Unary right fold (E op ...) becomes (E1 op (... op (EN-1 op EN)))
 * 2) Unary left fold (... op E) becomes (((E1 op E2) op ...) op EN)
 * 3) Binary right fold (E op ... op I) becomes (E1 op (... op (EN−1 op (EN op I))))
 * 4) Binary left fold (I op ... op E) becomes ((((I op E1) op E2) op ...) op EN)
 * 
 * template <typename... E>
 * foo (E... e)
 * foo(1, 2, 3, 4)
 * 
 * 1. (e - ...)     ->         (1 - (2 - (3 - 4)))
 * 2. (... - e)     ->       (((1 - 2) - 3) - 4)
 * 3. (e - ... - 0) ->         (1 - (2 - (3 - (4 - 0))))
 * 4. (0 - ... - e) -> ((((0 - 1) - 2) - 3) - 4)
*/

// Returns whether they are all true.
// Both 1 and 2 work, but currently uses 1.
template <typename... Types>
bool all (Types... args) {
  return (args && ...);
}

// Uses #4
template <typename... Types>
void print_all (Types... args) {
  (std::cout << ... << args) << '\n';
}

template <typename... Types>
auto sum (Types... args) {
  return (args + ...);
}

/*
 * NON FOLDING
 */

// sum2 has 2 seperate functions for the recursion and the base case.
template <typename T>
auto sum2 (T first) {
  return first;
}

template <typename T, typename... Types>
auto sum2 (T first, Types... args) {
  return first + sum2(args...);
}

int main (int argc, char *argv[]) {
  print_all(1, 2);

  std::cout << sum(1, 3.3, 4, 5) << '\n';;

  return 0;
}
