#include <iostream>

// sum_recursion has 2 seperate functions for the recursion and the base case.
template <typename T>
auto sum_recursion (T first) {
  return first;
}

template <typename T, typename... Types>
auto sum_recursion (T first, Types... args) {
  // Args contains all but the first element.
  // When args contains only one element, the base case is called.
  return first + sum2(args...);
}

/*
 * FOLDING
 *
 * 1) Unary right fold (E op ...) becomes (E1 op (... op (EN-1 op EN)))
 * 2) Unary left fold (... op E) becomes (((E1 op E2) op ...) op EN)
 * 3) Binary right fold (E op ... op I) becomes (E1 op (... op (EN−1 op (EN op I))))
 * 4) Binary left fold (I op ... op E) becomes ((((I op E1) op E2) op ...) op EN)
 *
 * template <typename... E>
 * foo (E... e) {}
 * foo(1, 2, 3, 4)
 *
 * 1. (e - ...)     ->         (1 - (2 - (3 - 4)))
 * 2. (... - e)     ->       (((1 - 2) - 3) - 4)
 * 3. (e - ... - 0) ->         (1 - (2 - (3 - (4 - 0))))
 * 4. (0 - ... - e) -> ((((0 - 1) - 2) - 3) - 4)
 *
 * In all cases, the side of the ... far away from e is evaluated first, and e is evaluated last.
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
auto sum_fold (Types... args) {
  return (args + ...);
}

int main (int argc, char *argv[]) {
  print_all(1, 2);

  std::cout << sum_fold(1, 3.3, 4, 5) << '\n';;

  return 0;
}
