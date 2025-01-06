#include <iostream>
#include <array>

template <typename T, size_t rows, size_t cols>
using array_2d = std::array<std::array<T, cols>, rows>;

// Recursive case
template <typename T, size_t first_size, size_t... rest>
struct multidimensional_array_wrapper {
  using type = std::array<typename multidimensional_array_wrapper<T, rest...>::type, first_size>;
};

// Base case for 1D array.
template <typename T, size_t size>
struct multidimensional_array_wrapper<T, size> {
  using type = std::array<T, size>;
};

template <typename T, size_t... sizes>
using multidimensional_array = multidimensional_array_wrapper<T, sizes...>::type;

int main (int argc, char *argv[]) {
  const array_2d<int, 3, 4> my_array {};
  std::cout << my_array.size() << '\n';
  std::cout << my_array[0].size() << '\n';

  // In a c style array, the first number is the outermost size, and subsequent numbers go further inwards, with the last number being the sizes of the innermost arrays.
  int c_style_array[3][4] = {
    {0, 1, 2, 3},
    {4, 5, 6, 7},
    {8, 9, 10, 11}
  };
  for (size_t row {}; row < sizeof(c_style_array)/sizeof(c_style_array[0]); row++) {
    for (size_t col {}; col < sizeof(c_style_array[0])/sizeof(int); col++) {
      std::cout << c_style_array[row][col] << ", ";
    } std::cout << '\n';
  } std::cout << '\n';
  
  // Same goes for the multidimensional array, where the first size is the outermost size, and the last size is the size of the innermost arrays.
  // The one thing about using std::array is that all but the innermost arrays have to use 2 curly brackets instead of 1.
  // https://www.learncpp.com/cpp-tutorial/stdarray-of-class-types-and-brace-elision/
  multidimensional_array<int, 3, 4> array_3x4 = {{
      {0, 1, 2, 3},
      {4, 5, 6, 7},
      {8, 9, 10, 11},
    }};

  for (auto arr : array_3x4) {
    for (auto e : arr) {
      std::cout << e << ", ";
    } std::cout << '\n';
  } std::cout << '\n';

  // Again, everywhere but the innermost array needs to use double curly brackets.
  multidimensional_array<int, 2, 2, 2> array_2x2x2 = {{
      {{ {0, 1}, {2, 3} }},
      {{ {4, 5}, {6, 7} }}
    }};

  /* This would cause a syntax error:
     Even though the outermost array uses single brackets, the middle one doesn't.
  multidimensional_array<int, 2, 2, 2> array_2x2x23 = {{
      { {0, 1}, {2, 3} },
      { {4, 5}, {6, 7} }
    }};
  */

  for (auto arr_2x2 : array_2x2x2) {
    for (auto arr : arr_2x2) {
      for (auto e : arr) {
	std::cout << e << ", ";
      } std::cout << '\n';
    } std::cout << '\n';
  } std::cout << '\n';
  
  return 0;
}
