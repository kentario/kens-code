#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>

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

  const int BUFFER_SIZE = 1024;
  char write_buffer[BUFFER_SIZE];
  char read_buffer[BUFFER_SIZE];
  
  const int PORT_NUMBER = atoi(argv[2]);
  
  int client_socket_fd;
  if ((client_socket_fd = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
    error("Error openning socket");
  }
  
  struct hostent *server;
  
  server = gethostbyname(argv[1]);
  if (server == NULL) {
    error("Error, no such host");
  }

  struct sockaddr_in server_address;
  bzero(&server_address, sizeof(server_address));
  server_address.sin_family = AF_INET;
  bcopy((char *) server->h_addr, (char *) &server_address.sin_addr.s_addr, server->h_length);
  server_address.sin_port = htons(PORT_NUMBER);

  if (connect(client_socket_fd,
	      (struct sockaddr*) &server_address,
	      sizeof(server_address)) < 0) {
    error("Connection failed");
  }

  while (1) {
    bzero(write_buffer, BUFFER_SIZE);
    bzero(read_buffer, BUFFER_SIZE);
    
    printf("Type some stuff:\n");
    size_t read_this_much = read(0, write_buffer, BUFFER_SIZE);

    if (read_this_much < 0) {
      error("Error reading from user");
    }
      
    size_t thislen = strlen(write_buffer);
    if (write(client_socket_fd, write_buffer, thislen) < 0) {
      error("Error on writing");
    }

    read_this_much = read(client_socket_fd, read_buffer, BUFFER_SIZE);
    if (read_this_much < 0) {
      error("Error reading from server");
    }

    printf("Server: %s", read_buffer);

    if (!strcmp(write_buffer, "end\n")) break;
  }
    
  close(client_socket_fd);
  
  return 0;
}
