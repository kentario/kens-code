#include <iostream>
#include <iterator>

int main () {
  int a[] {1, 2, 3, 4};

  for (const auto &b : a) {
    std::cout << b << "\n";
  }
  
  return 0;
}
