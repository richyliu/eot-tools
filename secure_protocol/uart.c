#include "uart.h"
#include <stddef.h>

#define UART_BASE_ADDR(id) (0x40004000 + ((id) * 0x1000))

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

struct uart_handle {
    uart_regs_t *regs;
    int nonblocking;
};

static struct uart_handle uart_handles[5];

uart_handle_t* uart_init(uart_id_t uart_id) {
    if (uart_id > UART_4) {
        return NULL;
    }
    
    struct uart_handle *handle = &uart_handles[uart_id];
    handle->regs = (uart_regs_t *)UART_BASE_ADDR(uart_id);
    handle->nonblocking = 0;
    
    handle->regs->CTRL = 0;
    handle->regs->BAUDDIV = UART_BAUD_DIV;
    handle->regs->CTRL = UART_CTRL_TXEN | UART_CTRL_RXEN;
    
    return handle;
}

void uart_deinit(uart_handle_t *handle) {
    if (!handle || !handle->regs) {
        return;
    }
    handle->regs->CTRL = 0;
    handle->regs = NULL;
}

int uart_write_byte(uart_handle_t *handle, uint8_t byte) {
    if (!handle || !handle->regs) {
        return -1;
    }
    
    uart_regs_t *regs = handle->regs;
    
    while (regs->STATE & UART_STATE_TXFULL) {
        if (handle->nonblocking) {
            return -1;
        }
    }
    
    regs->DATA = byte;
    return 0;
}

int uart_read_byte(uart_handle_t *handle, uint8_t *byte) {
    if (!handle || !handle->regs || !byte) {
        return -1;
    }
    
    uart_regs_t *regs = handle->regs;
    
    if (!(regs->STATE & UART_STATE_RXFULL)) {
        if (handle->nonblocking) {
            return -1;
        }
        while (!(regs->STATE & UART_STATE_RXFULL)) {
        }
    }
    
    *byte = (uint8_t)(regs->DATA & 0xFF);
    return 0;
}

int uart_can_read(uart_handle_t *handle) {
    if (!handle || !handle->regs) {
        return 0;
    }
    return (handle->regs->STATE & UART_STATE_RXFULL) ? 1 : 0;
}

int uart_can_write(uart_handle_t *handle) {
    if (!handle || !handle->regs) {
        return 0;
    }
    return (handle->regs->STATE & UART_STATE_TXFULL) ? 0 : 1;
}

int uart_write(uart_handle_t *handle, const uint8_t *data, size_t len) {
    if (!handle || !data || len == 0) {
        return -1;
    }
    
    size_t written = 0;
    for (size_t i = 0; i < len; i++) {
        if (uart_write_byte(handle, data[i]) < 0) {
            if (handle->nonblocking) {
                break;
            }
        }
        written++;
    }
    
    return (int)written;
}

int uart_read(uart_handle_t *handle, uint8_t *buffer, size_t max_len) {
    if (!handle || !buffer || max_len == 0) {
        return -1;
    }
    
    size_t read_count = 0;
    for (size_t i = 0; i < max_len; i++) {
        if (uart_read_byte(handle, &buffer[i]) < 0) {
            break;
        }
        read_count++;
    }
    
    return (int)read_count;
}

void uart_set_nonblocking(uart_handle_t *handle, int enable) {
    if (handle) {
        handle->nonblocking = enable;
    }
}
