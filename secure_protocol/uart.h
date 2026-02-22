#ifndef UART_H
#define UART_H

#include <stdint.h>
#include <stddef.h>

typedef struct uart_handle uart_handle_t;

typedef enum {
    UART_0 = 0,
    UART_1 = 1,
    UART_2 = 2,
    UART_3 = 3,
    UART_4 = 4,
} uart_id_t;

uart_handle_t* uart_init(uart_id_t uart_id);
void uart_deinit(uart_handle_t *handle);

int uart_write_byte(uart_handle_t *handle, uint8_t byte);
int uart_read_byte(uart_handle_t *handle, uint8_t *byte);

int uart_can_read(uart_handle_t *handle);
int uart_can_write(uart_handle_t *handle);

int uart_write(uart_handle_t *handle, const uint8_t *data, size_t len);
int uart_read(uart_handle_t *handle, uint8_t *buffer, size_t max_len);

void uart_set_nonblocking(uart_handle_t *handle, int enable);

#endif
