/**
 * Common Communication Protocol Logic
 * Shared implementation for both Unix socket and ARM UART transports.
 * Only depends on the raw comm_* functions which are platform-specific.
 */

#include "comm.h"
#include <stddef.h>
#include <stdint.h>

#ifndef MAX_PKT_LEN
#define MAX_PKT_LEN 512
#endif

static int pkt_dropped[10];

void comm_send(communicator_t *comm, const session_id_t session_id,
               const msg_type_t msg_type, const uint8_t *msg,
               const size_t msg_len, const uint8_t *shared_secret) {
  static int pkt_ctr = 0;
  pkt_ctr++;
  for (int i = 0; i < (int)(sizeof(pkt_dropped) / sizeof(pkt_dropped[0]));
       i++) {
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
    if (!compute_hmac(shared_secret, buffer, total_len, signature)) {
      ext_io_eprintf("Failed to compute HMAC\n");
      ext_exit(1);
    }
    if (total_len + SIGNATURE_SIZE > MAX_PKT_LEN) {
      ext_io_eprintf("Message too long to send with signature (%u bytes)\n",
                     total_len + SIGNATURE_SIZE);
      ext_exit(1);
    }
    ext_memcpy(buffer + total_len, signature, SIGNATURE_SIZE);
    total_len += SIGNATURE_SIZE;
  }

  ext_io_printf(
      "[INFO] sending message of length %u (session_id=%u, msg_type=%d)",
      total_len, session_id, msg_type);
  ext_io_flush();
  for (int i = 0; i < (int)(total_len / 5); i++) {
    ext_timer_sleep_ms(5 * 15);
    ext_io_putc('.');
    ext_io_flush();
  }
  ext_io_puts(" sent\n");

  if (comm_send_raw(comm->comm_h, buffer, total_len) != (ssize_t)total_len) {
    ext_io_eprintf("Failed to send message\n");
    ext_exit(1);
  }
}

void comm_send_legacy(communicator_t *comm, const unit_id_t unit_id,
                      const uint8_t *msg, const size_t msg_len) {
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

ssize_t comm_recv(communicator_t *comm, session_id_t *session_id,
                  msg_type_t *msg_type, uint8_t *msg, const size_t max_msg_len,
                  const uint8_t *shared_secret) {
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
      ext_io_eprintf("Received legacy message too short (%zd bytes)\n",
                     recv_len);
      return -2;
    }
    size_t payload_len = (size_t)recv_len - 4;
    if (payload_len > max_msg_len) {
      ext_io_eprintf("Received legacy message too long (%u bytes)\n",
                     payload_len);
      ext_exit(1);
    }
    ext_memcpy(msg, buffer + 4, payload_len);
    return -((ssize_t)payload_len);
  }

  size_t payload_len = (size_t)recv_len - header_len;

  if (shared_secret != NULL) {
    if (recv_len < (ssize_t)(header_len + SIGNATURE_SIZE)) {
      ext_io_eprintf(
          "Received message too short for signature (data: %zd bytes)\n",
          recv_len);
      return -2;
    }
    uint8_t signature[SIGNATURE_SIZE];
    ext_memcpy(signature, buffer + recv_len - SIGNATURE_SIZE, SIGNATURE_SIZE);
    if (!verify_hmac(shared_secret, buffer, recv_len - SIGNATURE_SIZE,
                     signature)) {
      ext_io_eprintf("HMAC verification failed\n");
      return -2;
    }
    payload_len -= SIGNATURE_SIZE;
  }

  if (payload_len > max_msg_len) {
    ext_io_eprintf(
        "Received message too long (data: %u bytes, total: %zd bytes)\n",
        payload_len, recv_len);
    ext_exit(1);
  }

  ext_memcpy(session_id, buffer, sizeof(session_id_t));
  ext_memcpy(msg_type, buffer + sizeof(session_id_t), sizeof(msg_type_t));
  ext_memcpy(msg, buffer + header_len, payload_len);

  ext_io_printf(
      "[INFO] received message: session_id=%u, msg_type=%d, recv_len=%d\n",
      *session_id, *msg_type, recv_len);
  return (ssize_t)payload_len;
}

void init_communicator(communicator_t *comm, comm_device_type_t device_type,
                       const uint32_t timeout_ms) {
  comm->timeout_ms = timeout_ms;
  comm->comm_h = comm_init(device_type, timeout_ms);
  if (!comm->comm_h) {
    ext_io_eprintf("Failed to initialize communication\n");
    ext_exit(1);
  }
}

void add_drop_packet(int pkt_num) {
  ext_io_printf("[TEST] adding packet %d to drop list\n", pkt_num);

  for (size_t i = 0; i < sizeof(pkt_dropped) / sizeof(pkt_dropped[0]); i++) {
    if (pkt_dropped[i] == 0) {
      pkt_dropped[i] = pkt_num;
      return;
    }
  }

  ext_io_eprintf("Too many packets to drop, increase pkt_dropped array size\n");
  ext_exit(1);
}
