#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* output_string:  buffer to return encoded content
   output_alloc:  buffer has this many character allocated (includes space for null termination)
   input_string:  zero-terminated string to encode
   offset: the caesar-cypher offset
*/

int encode (char *output_string, size_t output_alloc, const char *input_string, int offset) {
  // Does not include space for null termination.
  size_t input_string_len = strlen(input_string);
  if (input_string_len + 3 > output_alloc - 1) {
    printf ("Input string is too large.");
    return -1;
  }

  bzero (output_string, output_alloc);
  
  sprintf(output_string, "%03d", offset);
  
  for (int i = 0; input_string[i]; i++) {
    unsigned char c = input_string[i] + offset;

    while (c > 127) {
      c -= 95;
    }

    output_string[i + 3] = c;
  }
  
  return 0;
}

/* output_string:  buffer to return decoded content
   output_alloc:  buffer has this many character allocated (includes space for null termination)
   input_string:  zero-terminated string to decode
   offset: the caesar-cypher offset
*/
int decode (char *output_string, size_t output_alloc, const char *input_string, int offset) {
  // Does not include space for null termination.
  size_t input_string_len = strlen(input_string);
  if (input_string_len - 3 > output_alloc - 1) {
    printf ("Input string is too large.");
    return -1;
  }

  bzero (output_string, output_alloc);
  
  sprintf(output_string, "%03d", offset);
  
  for (int i = 3; input_string[i]; i++) {
    unsigned char c = input_string[i] - offset;

    while (c < 32) {
      c += 95;
    }

    output_string[i - 3] = c;
  }
  
  return 0;
}

int main (int argc, char *argv[]) {
  if (argc != 3) {
    printf("The correct usage is %s <encode or decode> message\n", argv[0]);
    return EXIT_FAILURE;
  }
 
  // d for decode, e for encode.
  char action = argv[1][0];

  int input_length = sizeof(argv[1])/sizeof(argv[1][0]);
  
  // Plus 3 for the offset, plus 1 for null character.
  size_t result_alloc = input_length + 4;
  char result[result_alloc];

  int offset;
  
  switch (action) {
  case 'e':
    offset = (int) argv[2][0];
    
    encode(result, result_alloc, argv[2], offset);
    break;
  case 'd':
    char offset_string[] = {argv[2][0], argv[2][1], argv[2][2], '\0'};
    
    offset = atoi(offset_string);

    decode(result, result_alloc, argv[2], offset);
    break;
  default:
    printf("Please use either encode or decode for the second argument.\n");
    return EXIT_FAILURE;
  }
  
  printf("result: %s\n", result);
  return 0;
}
