/**
 * ARM Cortex-M4 bare metal implementation with QEMU semihosting for stdio.
 * Uses ARM semihosting to interface with QEMU's host I/O.
 */

#define NANOPRINTF_USE_FIELD_WIDTH_FORMAT_SPECIFIERS 1
#define NANOPRINTF_USE_PRECISION_FORMAT_SPECIFIERS 1
#define NANOPRINTF_USE_FLOAT_FORMAT_SPECIFIERS 0
#define NANOPRINTF_USE_LARGE_FORMAT_SPECIFIERS 0
#define NANOPRINTF_USE_SMALL_FORMAT_SPECIFIERS 1
#define NANOPRINTF_USE_BINARY_FORMAT_SPECIFIERS 0
#define NANOPRINTF_USE_WRITEBACK_FORMAT_SPECIFIERS 0
#define NANOPRINTF_USE_ALT_FORM_FLAG 1

#define NANOPRINTF_IMPLEMENTATION
#include "nanoprintf.h"
#include "ext_support.h"
#include <stddef.h>

/* ========== Simple string to number conversion ========== */

static long __strtol(const char *nptr, char **endptr, int base) {
    const char *s = nptr;
    unsigned long acc;
    int c;
    int neg = 0;

    if (base < 0 || base > 36) {
        if (endptr) *endptr = (char *)nptr;
        return 0;
    }

    while (1) {
        c = *s;
        if (c >= '0' && c <= '9' && c - '0' < base) {
            break;
        }
        if (c == '-') {
            neg = 1;
            s++;
            break;
        }
        if (c == '+') {
            s++;
            break;
        }
        if (c == '\0') {
            if (endptr) *endptr = (char *)nptr;
            return 0;
        }
        s++;
    }

    acc = 0;
    while (1) {
        c = *s;
        if (c >= '0' && c <= '9' && c - '0' < base) {
            acc = acc * (unsigned long)base + (unsigned long)(c - '0');
        } else {
            break;
        }
        s++;
    }

    if (endptr) *endptr = (char *)s;
    return neg ? -(long)acc : (long)acc;
}

static unsigned long __strtoul(const char *nptr, char **endptr, int base) {
    return (unsigned long)__strtol(nptr, endptr, base);
}

/* ========== ARM Semihosting ========== */

__attribute__((naked)) void __semihosting_char_write0(const char *str) {
    __asm__ volatile (
        "mov r2, r0\n"
        "mov r0, #4\n"
        "mov r1, r2\n"
        "bkpt 0xAB\n"
        "bx lr\n"
    );
}

__attribute__((naked)) void __semihosting_char_writec(const char *c) {
    __asm__ volatile (
        "mov r1, r0\n"
        "mov r0, #3\n"
        "bkpt 0xAB\n"
        "bx lr\n"
    );
}

__attribute__((naked)) int __semihosting_char_readc(void) {
    __asm__ volatile (
        "mov r0, #7\n"
        "bkpt 0xAB\n"
        "bx lr\n"
    );
}

__attribute__((naked)) void __semihosting_exit(int status) {
    __asm__ volatile (
        "mov r1, r0\n"
        "movw r0, #:lower16:0x20026\n"
        "movt r0, #:upper16:0x20026\n"
        "bkpt 0xAB\n"
        "bx lr\n"
    );
}

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

static void stdout_putc(int c, void *ctx) {
    (void)ctx;
    __semihosting_char_writec((char *)&c);
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
    __semihosting_char_writec(&c);
}

void ext_io_puts(const char *str) {
    __semihosting_char_write0(str);
}

void ext_io_flush(void) {
}

int ext_io_getc(void) {
    return __semihosting_char_readc();
}

static char input_buffer[128];
static char writec_buf;
static size_t input_pos = 0;
static size_t input_len = 0;

int ext_io_getline(char *buffer, size_t max_len) {
    if (max_len == 0) {
        return -1;
    }

    size_t i = 0;
    while (i < max_len - 1) {
        if (input_pos >= input_len) {
            input_len = 0;
            input_pos = 0;
            int c = __semihosting_char_readc();
            if (c < 0 || c == 4) {
                if (i == 0) {
                    return -1;
                }
                break;
            }
            if (c == '\r' || c == '\n') {
                writec_buf = '\r';
                __semihosting_char_writec(&writec_buf);
                writec_buf = '\n';
                __semihosting_char_writec(&writec_buf);
                break;
            }
            if (c >= 32 && c < 127) {
                input_buffer[input_len++] = (char)c;
                writec_buf = (char)c;
                __semihosting_char_writec(&writec_buf);
            } else if (c == 127 || c == 8) {
                if (input_len > 0) {
                    input_len--;
                    writec_buf = '\b';
                    __semihosting_char_writec(&writec_buf);
                    writec_buf = ' ';
                    __semihosting_char_writec(&writec_buf);
                    writec_buf = '\b';
                    __semihosting_char_writec(&writec_buf);
                }
            }
        }

        if (input_pos < input_len) {
            char c = input_buffer[input_pos++];
            if (c == '\r' || c == '\n') {
                break;
            }
            buffer[i++] = c;
        } else {
            break;
        }
    }

    buffer[i] = '\0';
    input_pos = 0;
    input_len = 0;
    return (int)i;
}

int ext_io_scan_int(int *value) {
    char buffer[32];
    if (ext_io_getline(buffer, sizeof(buffer)) < 0) {
        return -1;
    }

    char *endptr;
    long val = __strtol(buffer, &endptr, 10);

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
    unsigned long val = __strtoul(buffer, &endptr, 10);

    if (endptr == buffer || *endptr != '\0') {
        return -1;
    }

    *value = (unsigned int)val;
    return 0;
}

int ext_io_kbhit(void) {
    return 0;
}

void ext_io_clear_input(void) {
    input_pos = 0;
    input_len = 0;
}

void ext_io_set_nonblocking(int enable) {
    (void)enable;
}

static void stderr_putc(int c, void *ctx) {
    (void)ctx;
    char ch = (char)c;
    __semihosting_char_writec(&ch);
}

int ext_io_eprintf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = npf_vpprintf(stderr_putc, NULL, format, args);
    va_end(args);
    return ret;
}

void ext_exit(int status) {
    __semihosting_exit(status);
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
