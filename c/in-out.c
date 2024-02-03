// Prints out what the user inputs.

#include <stdio.h>
#include <unistd.h>

int
main (int ac, char *av[])
{
  while (1) {
    char buffer[100];
    ssize_t read_this_much = read(STDIN_FILENO, buffer, 99);
    sleep(2);
    ssize_t wrote_this_much = write(STDOUT_FILENO, buffer, read_this_much);

    //    buffer[read_this_much] = 0;
    //printf ("%s", buffer);
  }
  return 0;
}
