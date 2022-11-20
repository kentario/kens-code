#include <stdio.h>
#include <stdlib.h>

int main()
{
  int a, c;
  char numstr[2];

  a = getchar();
  
  numstr[0] = getchar();
  numstr[1] = 0;
  
  int n = atoi(numstr);
  
  if (a == 'e') {
    printf ("d");
    printf ("%d", n);
    
    while ((c = getchar()) != EOF) {
      c = c + n;
      if (c > 126)
	c = c - 95;
      printf ("%c", c);
    }
  } else if (a == 'd') {
    printf ("The decoded version is: ");
    
    while ((c = getchar()) != EOF) {
      c = c - n;
      if (c < 32)
	c = c + 95;
      printf ("%c", c);
    }
  }
  printf ("\n");
}
