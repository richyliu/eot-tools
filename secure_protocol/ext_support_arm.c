/**
 * ARM Cortex-M4 bare metal stub implementation of support abstraction layer.
 * All functions are stubs that do nothing - actual implementations
 * should be provided by the application for specific hardware.
 */

#include "ext_support.h"
#include <stddef.h>

/* ========== Timer ========== */

int ext_timer_init(void) {
    return 0;
}

void ext_timer_now(ext_timer_t *t) {
    if (t) {
        t->timestamp_ms = 0;
    }
}

int ext_timer_diff_ms(const ext_timer_t *end, const ext_timer_t *start) {
    if (!end || !start) {
        return 0;
    }
    return (int)(end->timestamp_ms - start->timestamp_ms);
}

void ext_timer_sleep_ms(uint32_t ms) {
    (void)ms;
}

void ext_timer_sleep_us(uint32_t us) {
    (void)us;
}

/* ========== IO ========== */

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

/* ========== Random ========== */

#define NONCE_SIZE 16

int ext_random_init(void) {
    return 0;
}

int ext_random_bytes(uint8_t *buffer, size_t len) {
    if (!buffer) {
        return -1;
    }
    for (size_t i = 0; i < len; i++) {
        buffer[i] = 0;
    }
    return 0;
}

uint32_t ext_random_u32(void) {
    return 0;
}

uint32_t ext_random_range(uint32_t min, uint32_t max) {
    if (min >= max) {
        return min;
    }
    return min;
}

int ext_random_nonce(uint8_t *nonce) {
    if (!nonce) {
        return -1;
    }
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
