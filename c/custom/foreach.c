#include <stdio.h>

#define foreach(index, Type, element, arr) Type element = (arr)[0]; \
for (int index = 0; \
index < sizeof(arr)/sizeof((arr)[0]); \
index++, element = (arr)[index])

int main (int argc, char *argv[]) {
  double my_cool_constants[] = {1.414, 2.718, 3.141};

  foreach(i, double, cool_constant, my_cool_constants) {
      printf("index: %d, element: %f\n", i, cool_constant);
  }

  return 0;
};
