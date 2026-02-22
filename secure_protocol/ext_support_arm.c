/**
 * ARM Cortex-M4 bare metal implementation using UART for stdio.
 * Uses UART1 for I/O (UART0 is for device communication).
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
#include "uart.h"
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

static uint32_t systick_reload = 0;
static uint32_t systick_tick_hz = 0;
static uint32_t systick_ms_accum = 0;
static uint32_t systick_last_cvr = 0;

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

static void systick_poll_update(void) {
    uint32_t now = SYSTICK->CVR;
    uint32_t period = systick_reload + 1;

    uint32_t elapsed = (systick_last_cvr >= now) ? (systick_last_cvr - now) : (systick_last_cvr + period - now);
    systick_last_cvr = now;

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

/* ========== UART1 for I/O ========== */

static uart_handle_t *stdio_uart = NULL;
static int io_nonblocking = 0;

int ext_io_init(void) {
    stdio_uart = uart_init(UART_1);
    if (!stdio_uart) {
        return -1;
    }
    return 0;
}

static void stdout_putc(int c, void *ctx) {
    (void)ctx;
    if (stdio_uart) {
        uart_write_byte(stdio_uart, (uint8_t)c);
    }
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
    if (stdio_uart) {
        uart_write_byte(stdio_uart, (uint8_t)c);
    }
}

void ext_io_puts(const char *str) {
    if (!str || !stdio_uart) return;
    while (*str) {
        uart_write_byte(stdio_uart, (uint8_t)*str++);
    }
}

void ext_io_flush(void) {
}

int ext_io_getc(void) {
    if (!stdio_uart) return -1;
    
    uint8_t c;
    if (uart_read_byte(stdio_uart, &c) == 0) {
        return (int)c;
    }
    return -1;
}

static char input_buffer[128];
static size_t input_pos = 0;
static size_t input_len = 0;

int ext_io_getline(char *buffer, size_t max_len) {
    if (max_len == 0 || !stdio_uart) {
        return -1;
    }

    size_t i = 0;
    while (i < max_len - 1) {
        if (input_pos >= input_len) {
            input_len = 0;
            input_pos = 0;
            
            if (io_nonblocking && !uart_can_read(stdio_uart)) {
                if (i == 0) {
                    return -1;
                }
                break;
            }
            
            int c = ext_io_getc();
            if (c < 0 || c == 4) {
                if (i == 0) {
                    return -1;
                }
                break;
            }
            if (c == '\r' || c == '\n') {
                ext_io_putc('\r');
                ext_io_putc('\n');
                break;
            }
            if (c >= 32 && c < 127) {
                input_buffer[input_len++] = (char)c;
                ext_io_putc((char)c);
            } else if (c == 127 || c == 8) {
                if (input_len > 0) {
                    input_len--;
                    ext_io_putc('\b');
                    ext_io_putc(' ');
                    ext_io_putc('\b');
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
    if (!stdio_uart) return 0;
    return uart_can_read(stdio_uart);
}

void ext_io_clear_input(void) {
    input_pos = 0;
    input_len = 0;
}

void ext_io_set_nonblocking(int enable) {
    io_nonblocking = enable;
    if (stdio_uart) {
        uart_set_nonblocking(stdio_uart, enable);
    }
}

static void stderr_putc(int c, void *ctx) {
    (void)ctx;
    if (stdio_uart) {
        uart_write_byte(stdio_uart, (uint8_t)c);
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
    ext_io_printf("Exiting with status %d\n", status);
    while (1) { }
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

int uECC_RNG_func(uint8_t *dest, unsigned size) {
  return ext_random_bytes(dest, size) == 0;
}

int ext_random_init(uint32_t seed) {
    uECC_set_rng(uECC_RNG_func);
    
    if (seed != 0) {
        prng_state = seed;
    } else {
        prng_state = systick_get_ms() ^ 0xDEADBEEF;
    }

    ext_io_printf("[DEBUG] ext_random_init with prng_state=%08x\n", prng_state);
    
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
