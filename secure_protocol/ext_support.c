/**
 * Unix implementation of support abstraction layer.
 * Combines IO, timer, random, and utility functions.
 */

#define NANOPRINTF_IMPLEMENTATION
#include "nanoprintf.h"
#include "ext_support.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>

/* ========== Timer ========== */

int ext_timer_init(void) {
    return 0;
}

void ext_timer_now(ext_timer_t *t) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t->timestamp_ms = (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

int ext_timer_diff_ms(const ext_timer_t *end, const ext_timer_t *start) {
    return (int)(end->timestamp_ms - start->timestamp_ms);
}

void ext_timer_sleep_ms(uint32_t ms) {
    usleep(ms * 1000);
}

/* ========== IO ========== */

static int stdin_flags = 0;

int ext_io_init(void) {
    stdin_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (stdin_flags < 0) {
        return -1;
    }
    return 0;
}

static void stdout_putc(int c, void *ctx) {
    (void)ctx;
    fputc(c, stdout);
}

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
    
    if (endptr == buffer || *endptr != '\0') {
        return -1;
    }
    
    *value = (int)val;
    return 0;
}

int ext_io_scan_uint(uint32_t *value) {
    char buffer[32];
    if (ext_io_getline(buffer, sizeof(buffer)) < 0) {
        return -1;
    }
    
    char *endptr;
    unsigned long val = strtoul(buffer, &endptr, 10);
    
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

/* ========== Random ========== */

#define NONCE_SIZE 16

static int urandom_fd = -1;

int ext_random_init(void) {
    if (urandom_fd < 0) {
        urandom_fd = open("/dev/urandom", O_RDONLY);
        if (urandom_fd < 0) {
            return -1;
        }
    }
    return 0;
}

int ext_random_bytes(uint8_t *buffer, size_t len) {
    if (urandom_fd < 0) {
        if (ext_random_init() < 0) {
            return -1;
        }
    }
    
    size_t total_read = 0;
    while (total_read < len) {
        ssize_t n = read(urandom_fd, buffer + total_read, len - total_read);
        if (n < 0) {
            return -1;
        }
        total_read += (size_t)n;
    }
    return 0;
}

uint32_t ext_random_u32(void) {
    uint32_t val;
    ext_random_bytes((uint8_t *)&val, sizeof(val));
    return val;
}

uint32_t ext_random_range(uint32_t min, uint32_t max) {
    if (min >= max) {
        return min;
    }
    
    uint32_t range = max - min + 1;
    uint32_t limit = (UINT32_MAX / range) * range;
    uint32_t val;
    
    do {
        val = ext_random_u32();
    } while (val >= limit);
    
    return min + (val % range);
}

int ext_random_nonce(uint8_t *nonce) {
    return ext_random_bytes(nonce, NONCE_SIZE);
}

/* ========== Utils ========== */

void *ext_memset(void *dest, int value, size_t n) {
    unsigned char *d = (unsigned char *)dest;
    unsigned char v = (unsigned char)value;
    while (n--) {
        *d++ = v;
    }
    return dest;
}

void *ext_memcpy(void *dest, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dest;
    const unsigned char *s = (const unsigned char *)src;
    while (n--) {
        *d++ = *s++;
    }
    return dest;
}

int ext_memcmp(const void *lhs, const void *rhs, size_t n) {
    const unsigned char *l = (const unsigned char *)lhs;
    const unsigned char *r = (const unsigned char *)rhs;
    while (n--) {
        if (*l != *r) {
            return *l - *r;
        }
        l++;
        r++;
    }
    return 0;
}
