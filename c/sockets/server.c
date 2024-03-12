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
    fprintf(stderr, "Usage: %s <PORT_NUMBER>\n", argv[0]);
    exit(1);
  }
  
  const int PORT_NUMBER = atoi(argv[1]);
  // Allow a maximum of 10 connections to the socket.
  const int MAX_CONNECTIONS = 10;

  // Socket creates an endpoint and returns the file descriptor to that opened endpoint.
  // AF_INET means the domain will be IPv4 internect protocalls.
  // SOCK_STREAM means TCP protocol.
  // 0 is the default TCP protocol.
  int server_socket_fd;
  // According to https://beej.us/guide/bgnet/html/
  // It is better to use PF_INET in the call to socket, and AF_INET in the struct, even though they are the same.
  if ((server_socket_fd = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
    error("Error openning socket");
  }

  struct sockaddr_in server_address;
  bzero(&server_address, sizeof(server_address));

  server_address.sin_family      = AF_INET;
  server_address.sin_addr.s_addr = htonl(INADDR_ANY);
  server_address.sin_port        = htons(PORT_NUMBER);

  // Bind assigns an address to a socket using its file descriptor.
  // Returns 0 on success, -1 on failure.
  if (bind(server_socket_fd, (struct sockaddr *) &server_address, sizeof(server_address)) < 0) {
    error("Binding error");
  }

  if (listen(server_socket_fd, MAX_CONNECTIONS) < 0) {
    error("Listening error");
  }

  struct sockaddr_storage client_address;
  socklen_t client_address_len = sizeof(client_address);

  printf("Waiting for connection on port %d\n", PORT_NUMBER);
  
  int connection_fd = accept(server_socket_fd,
			     (struct sockaddr *) &client_address,
			     &client_address_len);

  printf("Connection accepted\n");

  char buffer[1024];
  const int BUFFER_SIZE = 1024;
  
  while (1) {
    bzero(buffer, BUFFER_SIZE);

    if (read(connection_fd, buffer, strlen(buffer)) < 0) {
      error("Error on reading");
    }

    bzero(buffer, BUFFER_SIZE);
    snprintf(buffer, BUFFER_SIZE, "HTTP/1.0 200 OK\r\n\r\nHello Isaac.");
    write(connection_fd, buffer, strlen(buffer));
    
    // Temporary
    break;
    //
    
    if (write(connection_fd, buffer, strlen(buffer)) < 0) {
      error("Error on writing");
    }
    
    printf("%s", buffer);
  }

  close(connection_fd);
  close(server_socket_fd);
  
  return 0;
}
