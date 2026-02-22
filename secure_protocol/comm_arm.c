#include "comm.h"
#include "uart.h"
#include "ext_support.h"
#include <stddef.h>
#include <stdint.h>

struct comm_handle {
    uint32_t timeout_ms;
    uart_handle_t *uart;
    comm_device_type_t device_type;
};

static struct comm_handle static_handle;

comm_handle_t* comm_init(comm_device_type_t device_type, uint32_t timeout_ms) {
    uart_handle_t *uart = uart_init(UART_0);
    if (!uart) {
        return NULL;
    }
    
    static_handle.timeout_ms = timeout_ms;
    static_handle.uart = uart;
    static_handle.device_type = device_type;
    
    return &static_handle;
}

static int uart_read_byte_timeout(uart_handle_t *uart, uint8_t *byte, uint32_t timeout_ms) {
    ext_timer_t start, now;
    ext_timer_now(&start);
    
    while (!uart_can_read(uart)) {
        ext_timer_now(&now);
        if (ext_timer_diff_ms(&now, &start) >= (int)timeout_ms) {
            return -1;
        }
    }
    
    return uart_read_byte(uart, byte);
}

ssize_t comm_send_raw(comm_handle_t *handle, const uint8_t *data, size_t len) {
    if (!handle || !handle->uart || len > 0xFFFF) {
        return -1;
    }
    
    uart_handle_t *uart = handle->uart;
    
    uint16_t length = (uint16_t)len;
    uart_write_byte(uart, (uint8_t)(length & 0xFF));
    uart_write_byte(uart, (uint8_t)((length >> 8) & 0xFF));
    
    for (size_t i = 0; i < len; i++) {
        uart_write_byte(uart, data[i]);
    }
    
    return (ssize_t)len;
}

ssize_t comm_recv_raw(comm_handle_t *handle, uint8_t *buffer, size_t max_len) {
    if (!handle || !handle->uart || max_len < 2) {
        return -2;
    }
    
    uart_handle_t *uart = handle->uart;
    uint32_t timeout = handle->timeout_ms;
    
    uint8_t length_bytes[2];
    
    if (uart_read_byte_timeout(uart, &length_bytes[0], timeout) < 0) {
        ext_io_eprintf("Timeout waiting for length byte 1\n");
        return -1;
    }
    
    if (uart_read_byte_timeout(uart, &length_bytes[1], timeout) < 0) {
        ext_io_eprintf("Timeout waiting for length byte 2\n");
        return -1;
    }
    
    uint16_t length = (uint16_t)length_bytes[0] | ((uint16_t)length_bytes[1] << 8);
    
    if (length > max_len) {
        return -2;
    }
    
    for (uint16_t i = 0; i < length; i++) {
        if (uart_read_byte_timeout(uart, &buffer[i], timeout) < 0) {
            ext_io_eprintf("Timeout waiting for data byte %u\n", i);
            return -1;
        }
    }
    
    return (ssize_t)length;
}

void comm_close(comm_handle_t *handle) {
    if (!handle) {
        return;
    }
    
    if (handle->uart) {
        uart_deinit(handle->uart);
        handle->uart = NULL;
    }
}
