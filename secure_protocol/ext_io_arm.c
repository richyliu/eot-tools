/**
 * ARM Cortex-M4 bare metal stub implementation of IO abstraction layer.
 * All functions are stubs that do nothing - actual implementations
 * should be provided by the application for specific hardware.
 */

#include "ext_io.h"
#include <stddef.h>
#include <stdarg.h>

int ext_io_init(void) {
    return 0;
}

int ext_io_printf(const char *format, ...) {
    (void)format;
    return 0;
}

int ext_io_vprintf(const char *format, va_list args) {
    (void)format;
    (void)args;
    return 0;
}

int ext_io_snprintf(char *buffer, size_t bufsz, const char *format, ...) {
    (void)format;
    if (buffer && bufsz > 0) {
        buffer[0] = '\0';
    }
    return 0;
}

int ext_io_pprintf(ext_io_putc_fn putc, void *ctx, const char *format, ...) {
    (void)putc;
    (void)ctx;
    (void)format;
    return 0;
}

void ext_io_putc(char c) {
    (void)c;
}

void ext_io_puts(const char *str) {
    (void)str;
}

void ext_io_flush(void) {
}

int ext_io_getc(void) {
    return -1;
}

int ext_io_getline(char *buffer, size_t max_len) {
    (void)buffer;
    (void)max_len;
    return -1;
}

int ext_io_scan_int(int *value) {
    (void)value;
    return -1;
}

int ext_io_scan_uint(unsigned int *value) {
    (void)value;
    return -1;
}

int ext_io_kbhit(void) {
    return 0;
}

void ext_io_clear_input(void) {
}

void ext_io_set_nonblocking(int enable) {
    (void)enable;
}

int ext_io_eprintf(const char *format, ...) {
    (void)format;
    return 0;
}

void ext_exit(int status) {
    (void)status;
    while (1) { }
}
