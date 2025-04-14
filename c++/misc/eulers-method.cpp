#include <iostream>
#include <array>

double dy_dx (const double x, const double y) {
  return 2 * x + y;
}

template <size_t NUM_VALUES, double STEP_SIZE>
std::array<std::array<double, 2>, NUM_VALUES> eulers_method (const std::array<double, 2> initial_values) {
  std::array<std::array<double, 2>, NUM_VALUES> values {};
  values[0] = initial_values;

  for (size_t i {1}; i < values.size(); i++) {
    values[i][0] = values[i - 1][0] + STEP_SIZE;

    const double prev_x {values[i - 1][0]};
    const double prev_y {values[i - 1][1]};

    values[i][1] = values[i - 1][1] + dy_dx(prev_x, prev_y) * STEP_SIZE;
  }

  return values;
}

int main () {
  constexpr double step_size {0.5};
  constexpr int num_values {3};
  constexpr std::array<double, 2> initial_values {1, -3};

  for (auto value : eulers_method<num_values, step_size>(initial_values)) {
    std::cout << '(' << value[0] << ", " << value[1] << ")\n";
  }

  return 0;
}
