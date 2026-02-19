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
