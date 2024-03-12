#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/socket.h>
#include <sys/types.h>

#include <netinet/in.h>

void error (const char *msg) {
  perror(msg);
  exit(1);
}

int main (int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <HOSTNAME> <PORT_NUMBER>\n", argv[0]);
    exit(1);
  }

  const int PORT_NUMBER = atoi(argv[1]);

  int client_socket_fd;
  if ((client_socket_fd = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
    error("Error openning socket");
  }

  struct sockaddr_in server_address;
  bzero(&server_address, sizeof(server_address));
  
  return 0;
}
