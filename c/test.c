/* Print something */

#include <stdio.h>

int
main (int argc, char *argv[])
{
  if (argc < 2) {
    fprintf (stderr, "Usage:  test <string>\n");
    return -1;
  }

  int i;
  for (i=1; i<argc; i++) {
    printf ("Arg %d: %s\n", i, argv[i]);
  }
  
  printf ("%s\n", "hello");
  
  return 0;
}
