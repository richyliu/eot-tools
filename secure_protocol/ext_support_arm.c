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
#include "micro-ecc/uECC.h"
#include <stddef.h>

/* ========== Simple string to number conversion ========== */

static long __strtol(const char *nptr, char **endptr, int base, int *error) {
    const char *s = nptr;
    unsigned long acc;
    int c;
    int neg = 0;

    if (base < 0 || base > 36) {
        if (endptr) *endptr = (char *)nptr;
        *error = 1;
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
            *error = 1;
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

static unsigned long __strtoul(const char *nptr, char **endptr, int base, int *error) {
  return (unsigned long)__strtol(nptr, endptr, base, error);
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

/* ========== ARM SysTick Timer ========== */

#define SYSTICK_BASE 0xE000E010

typedef struct {
    volatile uint32_t CSR;
    volatile uint32_t RVR;
    volatile uint32_t CVR;
    volatile uint32_t CALIB;
} systick_t;

#define SYSTICK ((systick_t *)SYSTICK_BASE)

#define SYSTICK_CSR_ENABLE      (1u << 0)
#define SYSTICK_CSR_CLKSOURCE   (1u << 2)

#define SYSTICK_RVR_RELOAD_MASK 0x00FFFFFF

static uint32_t systick_reload = 0;        /* reload value (ticks-1) */
static uint32_t systick_tick_hz = 0;       /* ticks per second */
static uint32_t systick_ms_accum = 0;      /* accumulated ms */
static uint32_t systick_last_cvr = 0;      /* last sampled CVR */

static void systick_init(uint32_t cpu_hz, uint32_t tick_hz) {
    systick_tick_hz = cpu_hz;
    systick_reload = (cpu_hz / tick_hz) - 1;

    SYSTICK->CSR = 0;
    SYSTICK->RVR = systick_reload & SYSTICK_RVR_RELOAD_MASK;
    SYSTICK->CVR = 0;
    SYSTICK->CSR = SYSTICK_CSR_CLKSOURCE | SYSTICK_CSR_ENABLE;

    systick_last_cvr = SYSTICK->CVR;
    systick_ms_accum = 0;
}

/* Convert elapsed SysTick ticks into ms and accumulate. */
static void systick_poll_update(void) {
    uint32_t now = SYSTICK->CVR;
    uint32_t period = systick_reload + 1;

    /* elapsed ticks with wrap handling (down-counter) */
    uint32_t elapsed = (systick_last_cvr >= now) ? (systick_last_cvr - now) : (systick_last_cvr + period - now);
    systick_last_cvr = now;

    /* Convert ticks -> ms, accumulate remainder to avoid drift */
    static uint32_t tick_remainder = 0;
    tick_remainder += elapsed;

    uint32_t ticks_per_ms = systick_tick_hz / 1000;
    uint32_t ms_inc = tick_remainder / ticks_per_ms;
    tick_remainder -= ms_inc * ticks_per_ms;

    systick_ms_accum += ms_inc;
}

static uint32_t systick_get_ms(void) {
    systick_poll_update();
    return systick_ms_accum;
}

/* ========== Timer ========== */

int ext_timer_init(void) {
    uint32_t cpu_freq = 1000000;
    systick_init(cpu_freq, 1000);
    return 0;
}

void ext_timer_now(ext_timer_t *t) {
    if (t) {
        t->timestamp_ms = systick_get_ms();
    }
}

int ext_timer_diff_ms(const ext_timer_t *end, const ext_timer_t *start) {
    if (!end || !start) {
        return 0;
    }
    return (int)(end->timestamp_ms - start->timestamp_ms);
}

void ext_timer_sleep_ms(uint32_t ms) {
    uint32_t start = systick_get_ms();
    while ((systick_get_ms() - start) < ms) { }
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
    int error = 0;
    long val = __strtol(buffer, &endptr, 10, &error);
    if (error) {
        return -1;
    }

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
    int error = 0;
    unsigned long val = __strtoul(buffer, &endptr, 10, &error);
    if (error) {
        return -1;
    }

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
    ext_io_printf("Exiting with status %d\n", status);
    while (1) { }
}

/* ========== ARM Semihosting File Operations ========== */

#define SYS_OPEN  0x01
#define SYS_READ  0x06
#define SYS_CLOSE 0x02

typedef struct {
    void *handle;
} semihosting_file_t;

static int urandom_initialized = 0;

__attribute__((naked)) long __semihosting_syscall(int reason, void *arg) {
    __asm__ volatile (
        "mov r12, r0\n"
        "mov r0, r1\n"
        "mov r1, r2\n"
        "bkpt 0xAB\n"
        "bx lr\n"
    );
}

static long semihosting_call(int reason, void *arg) {
    register long r0 __asm__("r0") = reason;
    register void *r1 __asm__("r1") = arg;
    __asm__ volatile (
        "bkpt 0xAB"
        : "=r"(r0)
        : "r"(r0), "r"(r1)
    );
    return r0;
}

static int semihosting_open_read(const char *filename) {
    typedef struct {
        const char *name;
        int len;
        int mode;
    } open_arg_t;

    char filename_null[256];
    int len = 0;
    while (filename[len] && len < 255) {
        filename_null[len] = filename[len];
        len++;
    }
    filename_null[len] = '\0';

    open_arg_t arg = { filename_null, len, 0 };
    long result = semihosting_call(SYS_OPEN, &arg);
    if (result < 0) {
        return -1;
    }
    return (int)result;
}

static long semihosting_read(int fd, void *buffer, long size) {
    typedef struct {
        int fd;
        void *buffer;
        long size;
    } read_arg_t;

    read_arg_t arg = { fd, buffer, size };
    return semihosting_call(SYS_READ, &arg);
}

static long semihosting_close(int fd) {
    typedef struct {
        int fd;
    } close_arg_t;

    close_arg_t arg = { fd };
    return semihosting_call(SYS_CLOSE, &arg);
}

/* ========== xorshift32 PRNG ========== */

static uint32_t prng_state = 0;

static uint32_t xorshift32(void) {
    uint32_t x = prng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return prng_state = x;
}

/* ========== Random ========== */

#define NONCE_SIZE 16


// returns 1 on success, 0 on failure
int uECC_RNG_func(uint8_t *dest, unsigned size) {
  return ext_random_bytes(dest, size) == 0;
}

int ext_random_init(void) {
    if (urandom_initialized) {
        return 0;
    }

    uECC_set_rng(uECC_RNG_func);

    int fd = semihosting_open_read(":random");
    if (fd < 0) {
        ext_io_printf("failed to open :random, using fallback seed\n");
        prng_state = 12345;
        urandom_initialized = 1;
        return 0;
    }

    uint32_t seed;
    long bytes_read = semihosting_read(fd, &seed, sizeof(seed));
    semihosting_close(fd);

    if (bytes_read != sizeof(seed)) {
        prng_state = 12345;
    } else {
        prng_state = seed;
    }

    if (prng_state == 0) {
        prng_state = 12345;
    }

    urandom_initialized = 1;
    return 0;
}

int ext_random_bytes(uint8_t *buffer, size_t len) {
    if (!buffer) {
        return -1;
    }
    for (size_t i = 0; i < len; i++) {
        buffer[i] = (uint8_t)(xorshift32() & 0xFF);
    }
    return 0;
}

uint32_t ext_random_u32(void) {
    return xorshift32();
}

uint32_t ext_random_range(uint32_t min, uint32_t max) {
    if (min >= max) {
        return min;
    }
    uint32_t range = max - min + 1;
    return min + (xorshift32() % range);
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
