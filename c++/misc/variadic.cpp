#include <iostream>

template <typename... T>
auto test (T... args) {
  return (args | ...);
}

int main (int argc, char *argv[]) {
  std::cout << test(1, 2) << '\n';
  
  return 0;
}
