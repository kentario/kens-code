// Following tutorial https://www.youtube.com/playlist?list=PLvv0ScY6vfd_ocTP2ZLicgqKnvq50OCXM

#include <print>
#include <thread>
#include <iostream>

void test (int x) {
  std::println("Hello from thread!");
  std::println("Argument passed in: {}", x);
}

int main () {
  auto lambda = [](int x) {
    
  };
  
  std::thread my_thread (test, 100);
  // Wait for my_thread to finish.
  // If it isn't done when it goes out of scope, bad things happen.
  my_thread.join();
  
  std::println("hello from my main thread");

  return 0;
}

void count_to_x (size_t x) {
  for (volatile size_t i {0}; i < x; i++) {
  }
}

void compare_counting () {
  std::println("Starting non-threaded");
  count_to_x(1'000'000'000);
  count_to_x(1'000'000'000);
  count_to_x(1'000'000'000);
  count_to_x(1'000'000'000);
  count_to_x(1'000'000'000);
  std::println("Starting threaded");
  std::thread a (count_to_x, 1'000'000'000);
  std::thread b (count_to_x, 1'000'000'000);
  std::thread c (count_to_x, 1'000'000'000);
  std::thread d (count_to_x, 1'000'000'000);
  std::thread e (count_to_x, 1'000'000'000);
  a.join();
  b.join();
  c.join();
  d.join();
  e.join();
  std::println("ending threaded");
}
