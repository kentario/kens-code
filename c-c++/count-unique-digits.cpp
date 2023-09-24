#include <iostream>
#include <vector>
#include <algorithm>

std::vector<long> recursive_count (int n, int digits[], int current_digit, std::vector<long> &total_unique_digits) {
  for (digits[current_digit] = 0; digits[current_digit] < 10; digits[current_digit]++) {
    // If the current digit is the last digit, then start counting the number of unique digits.
    if (current_digit >= n - 1) {
      bool value_seen[10] {};
      int num_unique_digits {};

      for (int digit {0}; digit < n; digit++) {
	// For each digit, check if its value has been seen before.
	if (value_seen[digits[digit]]) {
	  // If the value has been seen, then go to the next digit.
	  continue;
	} else {
	  // If the value has not been seen yet, then set it to have been seen, and increment the number of unique digits.
	  value_seen[digits[digit]] = true;
	  num_unique_digits++;
	}
      }
      total_unique_digits[num_unique_digits]++;
    } else {
      // If the current digit is not the last digit, go to the next digit.
      recursive_count(n, digits, current_digit + 1, total_unique_digits);
    }
  }

  return total_unique_digits;
}

std::vector<long> brute_force_count (int n) {
  std::vector<long> total_unique_digits(n + 1, 0);
  int digits[n] {};

  recursive_count(n, digits, 0, total_unique_digits);

  return total_unique_digits;
}

long long pow (int x, int y) {
  long long result {1};
  for (int i = 0; i < y; i++) {
    result *= x;
  }

  return result;
}

long factorial (int x) {
  if (x == 1 || x == 0) {
    return 1;
  }
  return x * factorial(x - 1);
}

long multiply_and_apply (std::vector<int> elements, long (*function)(int)) {
  long product {1};

  for (const auto &element : elements) {
    product *= function(element);
  }
  
  return product;
}

long permutations_with_repetition (std::vector<int> n_vector) {
  // n is the sum of all elements of n_vector.
  // Returns n!/(The product of the factorials of every element in n_vector)

  int n {0};
  for (const auto &element : n_vector) {
    n += element;
  }

  long product {multiply_and_apply(n_vector, factorial)};
  
  return factorial(n)/product;
}

long equivalent_patterns (std::vector<int> n_vector) {
  long total_product {1};
  // For each string of duplicates, count the number of duplicates, and multiply total_product by the factorial.
  int num_duplicates {0};
  int previous_element {-1};
  
  for (const auto &element : n_vector) {
    // If the previous element is the same as the current element, then increment num_duplicates.
    if (element == previous_element) {
      num_duplicates++;
    } else {
      // If they are not the same, then multiply total_product by the factorial of num_duplicates, then set num_duplicates to 1.
      total_product *= factorial(num_duplicates);
      num_duplicates = 1;
    }
    
    previous_element = element;
  }

  // Multiply by the remaining duplicates.
  total_product *= factorial(num_duplicates);
  
  return total_product;
}

long permutations (int n, int r) {
  return factorial(n)/factorial(n - r);
}

std::vector<std::vector<int>> remove_duplicate_combinations (std::vector<std::vector<int>> &all_combinations) {
  std::vector<std::vector<int>> result;

  // For each combination.
  for (auto &combination : all_combinations) {
    // Sort the combination from lowest to highest.
    std::sort(combination.begin(), combination.end());

    bool is_duplicate {false};
    
    // Check if the combination has already been used.
    for (const auto &previous : result) {
      if (combination == previous) {
	// If the combination has already been used, then stop checking the rest of the results.
	is_duplicate = true;
	break;
      }
    }

    // If the combination was not a duplicate, then add it on to results.
    if (!is_duplicate) {
      result.push_back(combination);
    }
  }

  return result;
}

void generate_combinations (int n, int k, int current_sum, int current_index, std::vector<int> &current, std::vector<std::vector<int>> &result) {
  // If the index has managed to reach the end of the array (k), then check if the current sum is correct.
  if (current_index >= k) {
    // Check if the current sum is correct (n is the correct sum).
    if (current_sum == n) {
      // If it is correct, then add it to result.
      result.push_back(current);
    }
    // Go down another path.
    return;
  }

  // Try all possible values for the current index from 1 to (n - current_sum).
  for (int i {1}; i <= n - current_sum; i++) {
    current[current_index] = i;
    // Recursive call to the next index with the updated current_sum.
    generate_combinations(n, k, current_sum + i, current_index + 1, current, result);
  }
}

std::vector<std::vector<int>> generate_all_combinations (int n, int k) {
  std::vector<std::vector<int>> result;
  std::vector<int> current(k, 0);

  // generate_combinations will use current to make all possible combinations, and then keep pushing it back onto result.
  generate_combinations(n, k, 0, 0, current, result);

  // If there is something like {1, 1, 2} [1, 2, 1} {2, 1, 1} only keep {1, 1, 2}.
  result = remove_duplicate_combinations(result);
  
  return result;
}

std::vector<long long> math_count (const int n) {
  int max_types {};
  // max_types will hold the maximum number of unique digits that a n-digit number can have.
  // Since there can only be a maximum of 10 unique digits, max_types will be 10 or less.
  if (n > 10) {
    max_types = 10;
  } else {
    max_types = n;
  }
  
  // Size 6, initialized to 0.
  std::vector<long long> total_unique_digits(max_types + 1, 0);

  // k is the number of types (in this case the number of unique digits).
  for (int k {1}; k <= max_types; k++) {
    // Each vector that n_vector holds will hold the number of a certain type there are.
    // For example, for k = 3, one of the vectors might be {2, 2, 1}.
    // This means that there are 2 type 1s, 2 type 2s, and 1 type 3. This represents aabbc.
    // Another example for k = 3, would be {1, 1, 3}. 1 type 1, 1 type 2, and 3 type 3s. This represents abccc.
    // In all of these, the size of the vector is k, and the inside of the vector adds up to n.
    
    std::vector<std::vector<int>> n_vector_set = generate_all_combinations(n, k);

    for (const auto &n_vector : n_vector_set) {
      // For each n_vector, calculate the total number of patterns.
      long total_patterns {permutations_with_repetition(n_vector)};
      //      std::cout << "total_patterns for " << k << ": " << total_patterns << "\n";
      
      // Then divide the total number of patterns by the number of equivalent patterns to get the number of distinct patterns.
      long distinct_patterns {total_patterns/equivalent_patterns(n_vector)};
      //      std::cout << "distinct_patterns for " << k << ": " << distinct_patterns << "\n";
      
      // Then multiply the number of distinct patterns for each n_vector by 10 permute k (10!/(10 - k)!).
      long num_digit_sequences_for_k {permutations(10, k)};
      //      std::cout << "num_digit_sequences_for_k for " << k << ": " << num_digit_sequences_for_k << "\n";
      
      // Accumulate this number into the total_unique_digits vector.
      total_unique_digits[k] += distinct_patterns * num_digit_sequences_for_k;
    }
  }
  
  return total_unique_digits;
}

int main (int argc, char** argv) {
  // Don't do more than 18.
  if (argc != 2) {
    std::cout << "Please call this program in the format: " << argv[0] << " <num_digits>\n";
    return 1;
  }

  int num_digits {std::stoi(argv[1])};
  
  std::vector<long long> math_total_unique_digits = math_count(num_digits);
  long long num_numbers = pow(10, num_digits);

  long long total {};
  
  for (int i = 0; i < math_total_unique_digits.size(); i++) {
    total += math_total_unique_digits[i];
    std::cout << "mtud[i]: " << math_total_unique_digits[i] << "\n";
    std::cout << "static_cast<double>(mtud[i]): " << static_cast<long double>(math_total_unique_digits[i]) << "\n";
    std::cout << "static_cast<double>(mtud[i])/num_numbers: " << static_cast<long double>(math_total_unique_digits[i])/num_numbers << "\n";
    std::cout << "static_cast<double>(mtud[i])/num_numbers * 100: " << static_cast<long double>(math_total_unique_digits[i])/num_numbers * 100 << "\n";
    
    long double percentage = static_cast<long double>(math_total_unique_digits[i])/num_numbers * 100;
    
    std::cout << "For " << i << " unique digits, there are " << math_total_unique_digits[i] << " possibilities ";
    std::cout << "(" << percentage << "% chance).\n\n";
  }
  std::cout << "num_numbers: " << num_numbers << " total: " << total << "\n";

  return 0;
}
