/**
 * ARM Cortex-M4 bare metal stub implementation of communication layer.
 * All functions are stubs that do nothing - actual implementations
 * should use hardware communication interfaces (UART, SPI, etc.).
 */

#include "comm.h"
#include <stddef.h>
#include <stdint.h>

struct comm_handle {
    int dummy;
};

static struct comm_handle dummy_handle = {0};

comm_handle_t* comm_init(comm_device_type_t device_type, uint32_t timeout_ms) {
    (void)device_type;
    (void)timeout_ms;
    return &dummy_handle;
}

ssize_t comm_send_raw(comm_handle_t *handle, const uint8_t *data, size_t len) {
    (void)handle;
    (void)data;
    return (ssize_t)len;
}

ssize_t comm_recv_raw(comm_handle_t *handle, uint8_t *buffer, size_t max_len) {
    (void)handle;
    (void)buffer;
    (void)max_len;
    return -1;
}

void comm_close(comm_handle_t *handle) {
    (void)handle;
}

static int pkt_dropped[10];
static const int *pkt_dropped_ptr __attribute__((unused)) = pkt_dropped;

void comm_send(communicator_t *comm, const session_id_t session_id, const msg_type_t msg_type,
               const uint8_t *msg, const size_t msg_len, const uint8_t *shared_secret) {
    (void)comm;
    (void)session_id;
    (void)msg_type;
    (void)msg;
    (void)msg_len;
    (void)shared_secret;
}

void comm_send_legacy(communicator_t *comm, const unit_id_t unit_id, const uint8_t *msg, const size_t msg_len) {
    (void)comm;
    (void)unit_id;
    (void)msg;
    (void)msg_len;
}

ssize_t comm_recv(communicator_t *comm, session_id_t *session_id, msg_type_t *msg_type,
                  uint8_t *msg, const size_t max_msg_len, const uint8_t *shared_secret) {
    (void)comm;
    (void)session_id;
    (void)msg_type;
    (void)msg;
    (void)max_msg_len;
    (void)shared_secret;
    return -1;
}

void init_communicator(communicator_t *comm, comm_device_type_t device_type, const uint32_t timeout_ms) {
    (void)device_type;
    comm->timeout_ms = timeout_ms;
    comm->comm_h = comm_init(device_type, timeout_ms);
}

void add_drop_packet(int pkt_num) {
    (void)pkt_num;
}
