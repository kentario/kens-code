#include <iostream>
#include <string>
#include <vector>

#define PRINT(...) std::cout << #__VA_ARGS__ << "\n";

/*
1) Unary right fold (E op ...) becomes (E1 op (... op (EN-1 op EN)))
2) Unary left fold (... op E) becomes (((E1 op E2) op ...) op EN)
3) Binary right fold (E op ... op I) becomes (E1 op (... op (EN−1 op (EN op I))))
4) Binary left fold (I op ... op E) becomes ((((I op E1) op E2) op ...) op EN)

template <typename... E>
foo (E... e)
foo(1, 2, 3, 4)

1. (e - ...)     ->         (1 - (2 - (3 - 4)))
2. (... - e)     ->       (((1 - 2) - 3) - 4)
3. (e - ... - 0) ->         (1 - (2 - (3 - (4 - 0))))
4. (0 - ... - e) -> ((((0 - 1) - 2) - 3) - 4)
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

int main (int argc, char *argv[]) {
  print_all(1, 2);

  std::vector<std::string> v {"first", "second", "third"};
  
  std::cout << sum(v[0], " ", v[1], " hi ", v[2]) << '\n';

  return 0;
}
