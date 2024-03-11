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
  if (argc != 2) {
    error("Correct usage is ./server.c <port_number>");
  }

  const int port_number = atoi(argv[1]);

  // Socket creates an endpoint and returns the file descriptor to that opened endpoint.
  // AF_INET means the domain will be IPv4 internect protocalls.
  // SOCK_STREAM means TCP protocol.
  // 0 is the default tcp protocol.
  int server_socket_fd;
  if ((server_socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    error("Error openning socket.");
  }

  struct sockaddr_in server_address;
  bzero(&server_address, sizeof(server_address));

  server_address.sin_family      = AF_INET;
  server_address.sin_addr.s_addr = htonl(INADDR_ANY);
  server_address.sin_port        = htons(port_number);

  // Bind assigns an address to a socket using its file descriptor.
  // Returns 0 on success, -1 on failure.
  if (bind(server_socket_fd, (struct sockaddr *) &server_address, sizeof(server_address)) < 0) {
    error("Binding failed.");
  }
  
  return 0;
}
