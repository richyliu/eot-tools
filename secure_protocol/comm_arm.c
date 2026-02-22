/**
 * ARM Cortex-M4 UART Implementation for MPS2-AN386
 * Uses CMSDK UART at 0x40004000 for serial communication
 * Implements framing protocol: 2-byte length prefix (LE) + payload
 */

#include "comm.h"
#include "ext_support.h"
#include <stddef.h>
#include <stdint.h>

#define UART0_BASE 0x40004000

typedef struct {
    volatile uint32_t DATA;
    volatile uint32_t STATE;
    volatile uint32_t CTRL;
    volatile uint32_t INTSTATUS;
    volatile uint32_t BAUDDIV;
    volatile uint32_t RESERVED[3];
} uart_regs_t;

#define UART_STATE_TXFULL (1 << 0)
#define UART_STATE_RXFULL (1 << 1)
#define UART_CTRL_TXEN    (1 << 0)
#define UART_CTRL_RXEN    (1 << 1)

#define UART_BAUD_DIV 1625

struct comm_handle {
    uint32_t timeout_ms;
    uart_regs_t *uart;
    comm_device_type_t device_type;
};

static struct comm_handle static_handle;

static uart_regs_t *get_uart_regs(void) {
    return (uart_regs_t *)UART0_BASE;
}

comm_handle_t* comm_init(comm_device_type_t device_type, uint32_t timeout_ms) {
    uart_regs_t *uart = get_uart_regs();
    
    uart->CTRL = 0;
    uart->BAUDDIV = UART_BAUD_DIV;
    uart->CTRL = UART_CTRL_TXEN | UART_CTRL_RXEN;
    
    static_handle.timeout_ms = timeout_ms;
    static_handle.uart = uart;
    static_handle.device_type = device_type;
    
    return &static_handle;
}

static void uart_write_byte(uart_regs_t *uart, uint8_t byte) {
    while (uart->STATE & UART_STATE_TXFULL) {
    }
    uart->DATA = byte;
}

static int uart_read_byte_timeout(uart_regs_t *uart, uint8_t *byte, uint32_t timeout_ms) {
    ext_timer_t start, now;
    ext_timer_now(&start);
    
    while (!(uart->STATE & UART_STATE_RXFULL)) {
        ext_timer_now(&now);
        if (ext_timer_diff_ms(&now, &start) >= (int)timeout_ms) {
            return -1;
        }
    }
    
    *byte = (uint8_t)(uart->DATA & 0xFF);
    return 0;
}

ssize_t comm_send_raw(comm_handle_t *handle, const uint8_t *data, size_t len) {
    if (!handle || len > 0xFFFF) {
        return -1;
    }
    
    uart_regs_t *uart = handle->uart;
    
    uint16_t length = (uint16_t)len;
    uart_write_byte(uart, (uint8_t)(length & 0xFF));
    uart_write_byte(uart, (uint8_t)((length >> 8) & 0xFF));
    
    for (size_t i = 0; i < len; i++) {
        uart_write_byte(uart, data[i]);
    }
    
    return (ssize_t)len;
}

ssize_t comm_recv_raw(comm_handle_t *handle, uint8_t *buffer, size_t max_len) {
    if (!handle || max_len < 2) {
        return -2;
    }
    
    uart_regs_t *uart = handle->uart;
    uint32_t timeout = handle->timeout_ms;
    
    uint8_t length_bytes[2];
    
    if (uart_read_byte_timeout(uart, &length_bytes[0], timeout) < 0) {
        return -1;
    }
    
    if (uart_read_byte_timeout(uart, &length_bytes[1], timeout) < 0) {
        return -1;
    }
    
    uint16_t length = (uint16_t)length_bytes[0] | ((uint16_t)length_bytes[1] << 8);
    
    if (length > max_len) {
        return -2;
    }
    
    for (uint16_t i = 0; i < length; i++) {
        if (uart_read_byte_timeout(uart, &buffer[i], timeout) < 0) {
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
        handle->uart->CTRL = 0;
    }
}
