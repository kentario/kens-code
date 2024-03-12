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

  const int BUFFER_SIZE = 1024;
  char buffer[BUFFER_SIZE];
  
  const int PORT_NUMBER = atoi(argv[1]);

  const int LISTEN_BACKLOG = 10;
  
  // Allow a maximum of 10 connections to the socket.
  const int MAX_CONNECTIONS = 10;
  int num_connections = 0;
  int client_fds[MAX_CONNECTIONS];
  bzero(client_fds, sizeof(client_fds));

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

  int opt = 1;

  // Set the reuseaddr option of the server socket to true.
  if (setsockopt(server_socket_fd, SOL_SOCKET, SO_REUSEADDR, (char *) &opt, sizeof(opt)) < 0) {   
    error("setsockopt");
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

  if (listen(server_socket_fd, LISTEN_BACKLOG) < 0) {
    error("Listening error");
  }

  printf("Waiting for connection on port %d\n", PORT_NUMBER);

  struct sockaddr_storage client_address;
  socklen_t client_address_len = sizeof(client_address);
  
  client_fds[num_connections] = accept(server_socket_fd,
					   (struct sockaddr *) &client_address,
					   &client_address_len);
  if (client_fds[num_connections] < 0) {
    error("Accept failure");
  }
  
  num_connections++;
  
  printf("Connection accepted\n");

  while (1) {
    bzero(buffer, BUFFER_SIZE);
    int n = read(client_fds[num_connections - 1], buffer, BUFFER_SIZE);
    if (n < 0) {
      error("Error on reading");
    }

    printf("Client: %s\n", buffer);
    
    if (write(client_fds[num_connections - 1], buffer, strlen(buffer)) < 0) {
      error("Error on writing");
    }

    if (!strcmp(buffer, "end\n")) break;
  }

  for (int i = 0; i < num_connections; i++) {
    if (client_fds[i] > 0) {
      close(client_fds[i]);
    }
  }
  
  close(server_socket_fd);
  
  return 0;
}
