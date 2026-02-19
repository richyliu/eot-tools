/**
 * Unix implementation of IO abstraction layer.
 * Uses nanoprintf for formatted output and stdin/stdout for IO.
 * This file can be replaced with a bare metal UART implementation.
 */

#define NANOPRINTF_IMPLEMENTATION
#include "nanoprintf.h"
#include "ext_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>
#include <errno.h>

// File descriptor for stdin - stored to allow non-blocking operations
static int stdin_flags = 0;

int ext_io_init(void) {
    // Store original stdin flags for non-blocking toggle
    stdin_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (stdin_flags < 0) {
        return -1;
    }
    return 0;
}

// Callback for writing to stdout
static void stdout_putc(int c, void *ctx) {
    (void)ctx;
    fputc(c, stdout);
}

// Callback for writing to stderr
static void stderr_putc(int c, void *ctx) {
    (void)ctx;
    fputc(c, stderr);
}

int ext_io_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = npf_vpprintf(stdout_putc, NULL, format, args);
    va_end(args);
    return ret;
}

int ext_io_vprintf(const char *format, va_list args) {
    return npf_vpprintf(stdout_putc, NULL, format, args);
}

int ext_io_snprintf(char *buffer, size_t bufsz, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = npf_vsnprintf(buffer, bufsz, format, args);
    va_end(args);
    return ret;
}

int ext_io_pprintf(ext_io_putc_fn putc, void *ctx, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = npf_vpprintf(putc, ctx, format, args);
    va_end(args);
    return ret;
}

void ext_io_putc(char c) {
    fputc(c, stdout);
}

void ext_io_puts(const char *str) {
    fputs(str, stdout);
}

void ext_io_flush(void) {
    fflush(stdout);
    fflush(stderr);
}

int ext_io_getc(void) {
    return fgetc(stdin);
}

int ext_io_getline(char *buffer, size_t max_len) {
    if (max_len == 0) {
        return -1;
    }
    
    size_t i = 0;
    int c;
    
    while (i < max_len - 1) {
        c = fgetc(stdin);
        if (c == EOF) {
            if (i == 0) {
                return -1;
            }
            break;
        }
        if (c == '\n') {
            break;
        }
        buffer[i++] = (char)c;
    }
    
    buffer[i] = '\0';
    return (int)i;
}

int ext_io_scan_int(int *value) {
    char buffer[32];
    if (ext_io_getline(buffer, sizeof(buffer)) < 0) {
        return -1;
    }
    
    char *endptr;
    long val = strtol(buffer, &endptr, 10);
    
    // Check for conversion errors
    if (endptr == buffer || *endptr != '\0') {
        return -1;
    }
    
    *value = (int)val;
    return 0;
}

int ext_io_scan_uint(unsigned int *value) {
    char buffer[32];
    if (ext_io_getline(buffer, sizeof(buffer)) < 0) {
        return -1;
    }
    
    char *endptr;
    unsigned long val = strtoul(buffer, &endptr, 10);
    
    // Check for conversion errors
    if (endptr == buffer || *endptr != '\0') {
        return -1;
    }
    
    *value = (unsigned int)val;
    return 0;
}

int ext_io_kbhit(void) {
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    
    int c = fgetc(stdin);
    
    fcntl(STDIN_FILENO, F_SETFL, flags);
    
    if (c != EOF) {
        // Put the character back
        ungetc(c, stdin);
        return 1;
    }
    
    return 0;
}

void ext_io_clear_input(void) {
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    
    char buf[16];
    while (read(STDIN_FILENO, buf, sizeof(buf)) > 0) {
        // Discard
    }
    
    fcntl(STDIN_FILENO, F_SETFL, flags);
}

void ext_io_set_nonblocking(int enable) {
    if (enable) {
        fcntl(STDIN_FILENO, F_SETFL, stdin_flags | O_NONBLOCK);
    } else {
        fcntl(STDIN_FILENO, F_SETFL, stdin_flags);
    }
}

int ext_io_eprintf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = npf_vpprintf(stderr_putc, NULL, format, args);
    va_end(args);
    return ret;
}

void ext_exit(int status) {
    exit(status);
}
