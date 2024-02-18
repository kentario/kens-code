#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) {
  if (argc != 3) {
    printf("The correct usage is %s <encode or decode> message\n", argv[0]);
    return EXIT_FAILURE;
  }

  // d for decode, e for encode.
  char action = argv[1][0];

  int input_length = sizeof(argv[1])/sizeof(argv[1][0]);
  
  // Plus 3 for the offset, plus 1 for null character.
  char result[input_length + 4];

  int offset;
  
  switch (action) {
  case 'e':
    offset = argv[2][0];

    sprintf(result, "%03d", offset);

    // i starts at 3 to skip the offset.
    for (int i = 0; argv[2][i]; i++) {
      unsigned char c = argv[2][i] + offset;

      // Make sure the character stays below 127.
      while (c > 127) {
        c -= 95;
      }
      
      result[i + 3] = c;
    }

    result[input_length + 4] = '\0';
    break;
  case 'd':
    char offset_string[] = {argv[2][0], argv[2][1], argv[2][2], '\0'};
    offset = atoi(offset_string);

    // i starts at 3 to skip the offset.
    for (int i = 3; argv[2][i]; i++) {
      unsigned char c = argv[2][i] - offset;
      
      // Make sure the character stays above 32.
      while (c < 32) {
        c += 95;
      }

      result[i - 3] = c;
    }

    result[input_length - 3] = '\0';
    break;
  default:
    printf("Please use either encode or decode for the second argument.\n");
    return EXIT_FAILURE;
  }
  
  printf("result: %s\n", result);
  return 0;
}
