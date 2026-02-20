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

#define EOT_TO_HOT_SOCKET_PATH "/tmp/eot_to_hot.sock"
#define HOT_TO_EOT_SOCKET_PATH "/tmp/hot_to_eot.sock"

// Maximum packet length
#define MAX_PKT_LEN 512

struct comm_handle {
  int send_fd;
  int recv_fd;
  struct sockaddr_un send_addr;
  uint32_t timeout_ms;
};

comm_handle_t* comm_init(comm_device_type_t device_type, uint32_t timeout_ms) {
  comm_handle_t *handle = malloc(sizeof(comm_handle_t));
  if (!handle) {
    return NULL;
  }
  
  handle->timeout_ms = timeout_ms;

  const char *send_path;
  const char *recv_path;
  if (device_type == COMM_DEVICE_EOT) {
    send_path = EOT_TO_HOT_SOCKET_PATH;
    recv_path = HOT_TO_EOT_SOCKET_PATH;
  } else {
    send_path = HOT_TO_EOT_SOCKET_PATH;
    recv_path = EOT_TO_HOT_SOCKET_PATH;
  }
  
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

static int pkt_dropped[10];

void comm_send(communicator_t *comm, const session_id_t session_id, const msg_type_t msg_type,
               const uint8_t *msg, const size_t msg_len, const uint8_t *shared_secret) {
    static int pkt_ctr = 0;
    pkt_ctr++;
    for (int i = 0; i < (int)(sizeof(pkt_dropped) / sizeof(pkt_dropped[0])); i++) {
        if (pkt_dropped[i] == pkt_ctr) {
            ext_io_printf("[WARN] dropping packet %d for testing\n", pkt_ctr);
            return;
        }
    }

    size_t total_len = sizeof(session_id_t) + sizeof(msg_type_t) + msg_len;
    if (total_len > MAX_PKT_LEN) {
        ext_io_eprintf("Message too long to send (%u bytes)\n", total_len);
        ext_exit(1);
    }
    uint8_t buffer[MAX_PKT_LEN];

    ext_memcpy(buffer, &session_id, sizeof(session_id_t));
    ext_memcpy(buffer + sizeof(session_id_t), &msg_type, sizeof(msg_type_t));
    ext_memcpy(buffer + sizeof(session_id_t) + sizeof(msg_type_t), msg, msg_len);

    if (shared_secret != NULL) {
        uint8_t signature[SIGNATURE_SIZE];
        if (!sign_message(shared_secret, buffer, total_len, signature)) {
            ext_io_eprintf("Failed to sign message\n");
            ext_exit(1);
        }
        if (total_len + SIGNATURE_SIZE > MAX_PKT_LEN) {
            ext_io_eprintf("Message too long to send with signature (%u bytes)\n", total_len + SIGNATURE_SIZE);
            ext_exit(1);
        }
        ext_memcpy(buffer + total_len, signature, SIGNATURE_SIZE);
        total_len += SIGNATURE_SIZE;
    }

    ext_io_printf("[INFO] sending message of length %u (session_id=%u, msg_type=%d)", total_len, session_id, msg_type);
    ext_io_flush();
    for (int i = 0; i < (int)(total_len / 5); i++) {
        ext_timer_sleep_us(5 * 15 * 1000);
        ext_io_putc('.');
        ext_io_flush();
    }
    ext_io_puts(" sent\n");

    if (comm_send_raw(comm->comm_h, buffer, total_len) != (ssize_t)total_len) {
        ext_io_eprintf("Failed to send message\n");
        ext_exit(1);
    }
}

void comm_send_legacy(communicator_t *comm, const unit_id_t unit_id, const uint8_t *msg, const size_t msg_len) {
    uint32_t legacy_header;
    ext_memcpy(&legacy_header, "OLD!", 4);
    size_t total_len = sizeof(legacy_header) + sizeof(unit_id_t) + msg_len;
    if (total_len > MAX_PKT_LEN) {
        ext_io_eprintf("Message too long to send (%u bytes)\n", total_len);
        ext_exit(1);
    }
    uint8_t buffer[MAX_PKT_LEN];
    ext_memcpy(buffer, &legacy_header, sizeof(legacy_header));
    ext_memcpy(buffer + sizeof(legacy_header), &unit_id, sizeof(unit_id_t));
    ext_memcpy(buffer + sizeof(legacy_header) + sizeof(unit_id_t), msg, msg_len);
    if (comm_send_raw(comm->comm_h, buffer, total_len) != (ssize_t)total_len) {
        ext_io_eprintf("Failed to send legacy message\n");
        ext_exit(1);
    }
    ext_io_printf("sent legacy message of length %u\n", total_len);
}

ssize_t comm_recv(communicator_t *comm, session_id_t *session_id, msg_type_t *msg_type,
                  uint8_t *msg, const size_t max_msg_len, const uint8_t *shared_secret) {
    uint8_t buffer[MAX_PKT_LEN];
    size_t header_len = sizeof(session_id_t) + sizeof(msg_type_t);

    ssize_t recv_len = comm_recv_raw(comm->comm_h, buffer, sizeof(buffer));
    if (recv_len == -1) {
        return -1;
    }
    if (recv_len == -2) {
        ext_io_eprintf("IPC receive error\n");
        ext_exit(1);
    }

    if (recv_len >= 4 && ext_memcmp(buffer, "OLD!", 4) == 0) {
        if (recv_len < 4 + (ssize_t)sizeof(unit_id_t)) {
            ext_io_eprintf("Received legacy message too short (%zd bytes)\n", recv_len);
            return -2;
        }
        size_t payload_len = (size_t)recv_len - 4;
        if (payload_len > max_msg_len) {
            ext_io_eprintf("Received legacy message too long (%u bytes)\n", payload_len);
            ext_exit(1);
        }
        ext_memcpy(msg, buffer + 4, payload_len);
        return -((ssize_t)payload_len);
    }

    size_t payload_len = (size_t)recv_len - header_len;

    if (shared_secret != NULL) {
        if (recv_len < (ssize_t)(header_len + SIGNATURE_SIZE)) {
            ext_io_eprintf("Received message too short for signature (data: %zd bytes)\n", recv_len);
            return -2;
        }
        uint8_t signature[SIGNATURE_SIZE];
        ext_memcpy(signature, buffer + recv_len - SIGNATURE_SIZE, SIGNATURE_SIZE);
        if (!verify_signature(shared_secret, buffer, recv_len - SIGNATURE_SIZE, signature)) {
            ext_io_eprintf("Signature verification failed\n");
            return -2;
        }
        payload_len -= SIGNATURE_SIZE;
    }

    if (payload_len > max_msg_len) {
        ext_io_eprintf("Received message too long (data: %u bytes, total: %zd bytes)\n", payload_len, recv_len);
        ext_exit(1);
    }

    ext_memcpy(session_id, buffer, sizeof(session_id_t));
    ext_memcpy(msg_type, buffer + sizeof(session_id_t), sizeof(msg_type_t));
    ext_memcpy(msg, buffer + header_len, payload_len);

    return (ssize_t)payload_len;
}

void init_communicator(communicator_t *comm, comm_device_type_t device_type, const uint32_t timeout_ms) {
    comm->timeout_ms = timeout_ms;
    comm->comm_h = comm_init(device_type, timeout_ms);
    if (!comm->comm_h) {
        ext_io_eprintf("Failed to initialize communication\n");
        ext_exit(1);
    }
}

void add_drop_packet(int pkt_num) {
    for (size_t i = 0; i < sizeof(pkt_dropped) / sizeof(pkt_dropped[0]); i++) {
        if (pkt_dropped[i] == 0) {
            pkt_dropped[i] = pkt_num;
            return;
        }
    }

    ext_io_eprintf("Too many packets to drop, increase pkt_dropped array size\n");
    ext_exit(1);
}
