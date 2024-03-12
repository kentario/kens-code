#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <errno.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/time.h>

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

  fd_set read_fds;
  int num_active;
  
  FD_ZERO(&read_fds);
  FD_SET(client_socket_fd, &read_fds);
  // 0 is for stdin
  FD_SET(0, &read_fds);
  while (1) {
    // Wait for either the user to input something, or for the server to send a message.
    printf("waiting for activity\n");
    num_active = select(client_socket_fd + 1, &read_fds, NULL, NULL, NULL);
    printf("activity found\n");

    if ((num_active < 0) && (errno != EINTR)) {
      error("Error with select");
    }

    if (FD_ISSET(client_socket_fd, &read_fds)) {
      bzero(read_buffer, BUFFER_SIZE);
      
      // Read from server.
      ssize_t bytes_read = read(client_socket_fd, read_buffer, BUFFER_SIZE);

      if (bytes_read < 0) {
	error("Error reading from server");
      } else if (bytes_read == 0) {
	printf("Server closed");
	break;
      }
      
      printf("Server: %s", read_buffer);
    } else if (FD_ISSET(0, &read_fds)) {      
      bzero(write_buffer, BUFFER_SIZE);
      
      // Read from stdin.
      ssize_t bytes_read = read(0, write_buffer, BUFFER_SIZE);

      if (bytes_read < 0) {
	error("Error reading from user");
      }
      // Write to server.
      ssize_t bytes_written = write(client_socket_fd, write_buffer, strlen(write_buffer));

      if (bytes_written != strlen(write_buffer)) {
	error("Error writing to server");
      }

      if (!strcmp(write_buffer, "end\n")) break;
    }
  }
    
  close(client_socket_fd);
  
  return 0;
}
