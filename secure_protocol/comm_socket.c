/**
 * Socket-based Device Communication Implementation
 * Uses Unix domain sockets (SOCK_DGRAM) for communication
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <sys/time.h>

#include "comm.h"

// Maximum packet length
#define MAX_PKT_LEN 512

struct comm_handle {
  int send_fd;
  int recv_fd;
  struct sockaddr_un send_addr;
  uint32_t timeout_ms;
};

comm_handle_t* comm_init(const char *send_path, const char *recv_path, uint32_t timeout_ms) {
  comm_handle_t *handle = malloc(sizeof(comm_handle_t));
  if (!handle) {
    return NULL;
  }
  
  handle->timeout_ms = timeout_ms;
  
  // Create send socket
  handle->send_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
  if (handle->send_fd < 0) {
    perror("socket send");
    free(handle);
    return NULL;
  }
  
  memset(&handle->send_addr, 0, sizeof(handle->send_addr));
  handle->send_addr.sun_family = AF_UNIX;
  strncpy(handle->send_addr.sun_path, send_path, sizeof(handle->send_addr.sun_path) - 1);
  
  // Create recv socket
  handle->recv_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
  if (handle->recv_fd < 0) {
    perror("socket recv");
    close(handle->send_fd);
    free(handle);
    return NULL;
  }
  
  struct sockaddr_un recv_addr;
  memset(&recv_addr, 0, sizeof(recv_addr));
  recv_addr.sun_family = AF_UNIX;
  strncpy(recv_addr.sun_path, recv_path, sizeof(recv_addr.sun_path) - 1);
  
  // Remove existing socket file
  unlink(recv_path);
  
  // Bind receive socket
  if (bind(handle->recv_fd, (struct sockaddr*)&recv_addr, sizeof(recv_addr)) < 0) {
    perror("bind recv");
    close(handle->send_fd);
    close(handle->recv_fd);
    free(handle);
    return NULL;
  }
  
  // Set receive timeout
  struct timeval tv;
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  if (setsockopt(handle->recv_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
    perror("setsockopt");
    close(handle->send_fd);
    close(handle->recv_fd);
    free(handle);
    return NULL;
  }
  
  return handle;
}

ssize_t comm_send_raw(comm_handle_t *handle, const uint8_t *data, size_t len) {
  if (!handle || len > MAX_PKT_LEN) {
    return -1;
  }
  
  ssize_t sent = sendto(handle->send_fd, data, len, 0,
                        (struct sockaddr*)&handle->send_addr, 
                        sizeof(handle->send_addr));
  if (sent < 0) {
    perror("sendto");
    return -1;
  }
  
  return sent;
}

ssize_t comm_recv_raw(comm_handle_t *handle, uint8_t *buffer, size_t max_len) {
  if (!handle) {
    return -2;
  }
  
  ssize_t recv_len = recvfrom(handle->recv_fd, buffer, max_len, 0, NULL, NULL);
  if (recv_len < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      // Timeout
      return -1;
    }
    perror("recvfrom");
    return -2;
  }
  
  return recv_len;
}

void comm_close(comm_handle_t *handle) {
  if (!handle) {
    return;
  }
  
  if (handle->send_fd >= 0) {
    close(handle->send_fd);
  }
  if (handle->recv_fd >= 0) {
    close(handle->recv_fd);
  }
  
  free(handle);
}
