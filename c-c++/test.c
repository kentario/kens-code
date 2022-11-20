/* Print something */

#include <stdio.h>

int
main (int ac, char *av[])
{
  if (ac < 2) {
    fprintf (stderr, "Usage:  test <string>\n");
    return -1;
  }

  int i;
  for (i=1; i<ac; i++) {
    printf ("Arg %d: %s\n", i, av[i]);
  }
  
  printf ("%s\n", "hello");
  
  return 0;
}
