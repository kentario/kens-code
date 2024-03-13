#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>

#include <arpa/inet.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>

#include <netinet/in.h>

void error (const char *msg) {
  perror(msg);
  exit(1);
}

struct client_info {
  int client_fd;

  struct sockaddr_in client_address;
};

int main (int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <PORT_NUMBER>\n", argv[0]);
    exit(1);
  }

  const int BUFFER_SIZE = 1024;
  char read_buffer[BUFFER_SIZE];
  char write_buffer[BUFFER_SIZE];

  const int PORT_NUMBER = atoi(argv[1]);

  const int LISTEN_BACKLOG = 10;

  // Allow a maximum of 10 connections to the socket.
  const int MAX_CONNECTIONS = 10;
  int num_connections = 0;
  struct client_info clients[MAX_CONNECTIONS];
  bzero(clients, sizeof(clients));

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

  fd_set read_fds;
  int max_fd;
  int num_active;
  while (1) {
    FD_ZERO(&read_fds);
    max_fd = -1;

    // Read from terminal input.
    FD_SET(fileno(stdin), &read_fds);
    
    // Only detect incoming connections if there is space in the client array.
    if (num_connections < MAX_CONNECTIONS) {
      FD_SET(server_socket_fd, &read_fds);

      // Find the largest used file descriptor
      max_fd = server_socket_fd;
    }

    for (int i = 0; i < MAX_CONNECTIONS; i++) {
      // If the client is active.
      if (clients[i].client_fd > 0) {
	FD_SET(clients[i].client_fd, &read_fds);
      }

      if (clients[i].client_fd > max_fd) max_fd = clients[i].client_fd;
    }

    if (max_fd < 0) {
      fprintf(stderr, "No file descriptors");
      exit(1);
    }

    // Wait indefinitely for something to happen on a file descriptor
    // Select will check every file descriptor up until max_fd.
    num_active = select(max_fd + 1, &read_fds, NULL, NULL, NULL);

    if ((num_active < 0) && (errno != EINTR)) {
      error("Error with select");
    }

    // If stdin was one of the active sockets, then handle user input.
    if (FD_ISSET(fileno(stdin), &read_fds)) {
      // Write to all clients.
      ssize_t bytes_read = read(fileno(stdin), write_buffer, BUFFER_SIZE);
      if (bytes_read < 0) {
	error("Error reading from user");
      }
      
      for (int i = 0; i < MAX_CONNECTIONS; i++) {
	if (clients[i].client_fd <= 0) continue;

	ssize_t bytes_written = write(clients[i].client_fd, write_buffer, strlen(write_buffer));
	if (bytes_written != strlen(write_buffer)) {
	  error("Error writing to client");
	}
      }
    }
    
    // If the server socket was one of the active sockets, then handle an incoming connection.
    if (FD_ISSET(server_socket_fd, &read_fds)) {
      struct client_info *client = &clients[num_connections];
      
      socklen_t client_address_len = sizeof(client->client_address);

      clients[num_connections].client_fd = accept(server_socket_fd,
					   (struct sockaddr *) &client->client_address,
					   &client_address_len);

      if (client->client_fd < 0) {
	error("Accept failure");
      }

      // Print out information about the client.
      printf("New connection on socket %d, ip of client is: %s, client is sending from port: %d\n",
	     client->client_fd,
	     inet_ntoa(client->client_address.sin_addr),
	     ntohs(client->client_address.sin_port));

      // Send a greeting message.
      const char *message = "Welcome!\n";
      ssize_t bytes_written = write(client->client_fd, message, strlen(message));

      if (bytes_written != strlen(message)) {
	error("Write error");
      }

      num_connections++;
      num_active--;
    }

    // If there is an action on some other file descriptor besides the server or stdin.
    if (num_active > 0) {
      for (int i = 0; i < MAX_CONNECTIONS; i++) {
	if (clients[i].client_fd <= 0) continue;

	if (FD_ISSET(clients[i].client_fd, &read_fds)) {
	  // If a client sent a message, or closed, then handle it.

	  bzero(read_buffer, BUFFER_SIZE);
	  ssize_t bytes_read = read(clients[i].client_fd, read_buffer, BUFFER_SIZE);
	  if (bytes_read == 0 || !strcmp(read_buffer, "end")) {
	    close(clients[i].client_fd);
	    clients[i].client_fd = 0;
	    // TODO: Handle this better.
	  } else if (bytes_read < 0) {
	    error("Read error");
	  } else {
	    printf("Client: %s", read_buffer);
	    // Send back to the client.
	    ssize_t bytes_written = write(clients[i].client_fd, read_buffer, strlen(read_buffer));
	    if (bytes_written != strlen(read_buffer)) {
	      error("Write error");
	    }
	  }
	}
      }
    }
  }

  for (int i = 0; i < num_connections; i++) {
    if (clients[i].client_fd > 0) {
      close(clients[i].client_fd);
    }
  }

  close(server_socket_fd);

  return 0;
}
