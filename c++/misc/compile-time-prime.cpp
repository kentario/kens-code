#include <iostream>
#include <fstream>

// value is true if N is not divisible by any number from D or below.
template <int N, int D>
struct not_divisible {
  static constexpr bool value {(N % D) && not_divisible<N, D - 1>::value};
};

// If D is 1, then value should be true.
template <int N>
struct not_divisible<N, 1> {
  static constexpr bool value {true};
};

// For the case that is prime<1> happens.
template <>
struct not_divisible<1, 0> {
  static constexpr bool value {true};
};

template <int N>
struct is_prime {
  static constexpr bool value {not_divisible<N, N - 1>::value};
};

template <int N>
void for_loop (const std::string &file_name) {
  // Execute the code.
  //  std::cout << N << " is a prime number: " << (is_prime<N>::value ? "true" : "false") << "\n";

  if constexpr (is_prime<N>::value) {
    std::ofstream my_file;
    my_file.open(file_name, std::ios::app);
    my_file << N << "\n";
    my_file.close();
  }
  
  constexpr int MIN_VALUE {1};
  
  if constexpr (N > MIN_VALUE) {
    for_loop<N - 1>(file_name);
  }
}

int main () {
  std::string file_name {"prime-numbers.txt"};
  
  // Clear the file.
  std::ofstream file;
  file.open(file_name, std::ios::trunc);
  file.close();

  for_loop<500>(file_name);
};
