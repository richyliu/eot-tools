/**
 * Support abstraction layer for the EOT/HOT protocol.
 * Combines IO, timer, random, and utility functions.
 * Can be swapped out for bare metal implementations at build time.
 */

#ifndef EXT_SUPPORT_H_INCLUDED
#define EXT_SUPPORT_H_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Timer ========== */

/**
 * Timer type - opaque structure representing a timestamp.
 * The actual implementation is defined in the .c file.
 */
typedef struct ext_timer_s {
    uint64_t timestamp_ms;
} ext_timer_t;

/**
 * Initialize the timer subsystem.
 * Should be called once before using other timer functions.
 * @return 0 on success, -1 on error
 */
int ext_timer_init(void);

/**
 * Get the current time.
 * @param t Pointer to timer struct to fill with current time
 */
void ext_timer_now(ext_timer_t *t);

/**
 * Calculate the difference between two timestamps in milliseconds.
 * @param end The later time
 * @param start The earlier time
 * @return Difference in milliseconds (end - start)
 */
int ext_timer_diff_ms(const ext_timer_t *end, const ext_timer_t *start);

/**
 * Sleep for a specified number of milliseconds.
 * @param ms Number of milliseconds to sleep
 */
void ext_timer_sleep_ms(uint32_t ms);

/* ========== IO ========== */

/**
 * Character output callback function type.
 * Used for custom output targets (UART, file, etc.)
 */
typedef void (*ext_io_putc_fn)(int c, void *ctx);

/**
 * Initialize the IO subsystem.
 * Should be called once before using other IO functions.
 * @return 0 on success, -1 on error
 */
int ext_io_init(void);

/**
 * Print a formatted string using nanoprintf to stdout.
 * @param format printf-style format string
 * @param ... variable arguments
 * @return Number of characters that would have been written
 */
int ext_io_printf(const char *format, ...);

/**
 * Print a formatted string using nanoprintf with va_list.
 * @param format printf-style format string
 * @param args variable argument list
 * @return Number of characters that would have been written
 */
int ext_io_vprintf(const char *format, va_list args);

/**
 * Print a formatted string to a buffer using nanoprintf.
 * @param buffer Output buffer
 * @param bufsz Size of output buffer
 * @param format printf-style format string
 * @param ... variable arguments
 * @return Number of characters that would have been written
 */
int ext_io_snprintf(char *buffer, size_t bufsz, const char *format, ...);

/**
 * Print a formatted string using a custom putc callback.
 * @param putc Output callback function
 * @param ctx Context pointer passed to putc callback
 * @param format printf-style format string
 * @param ... variable arguments
 * @return Number of characters that would have been written
 */
int ext_io_pprintf(ext_io_putc_fn putc, void *ctx, const char *format, ...);

/**
 * Print a single character to output.
 * @param c Character to print
 */
void ext_io_putc(char c);

/**
 * Print a null-terminated string to output.
 * @param str String to print
 */
void ext_io_puts(const char *str);

/**
 * Flush output buffers (if applicable).
 */
void ext_io_flush(void);

/**
 * Read a single character from input (blocking).
 * @return Character read, or -1 on error
 */
int ext_io_getc(void);

/**
 * Read a line from input up to max_len-1 characters.
 * The newline character is consumed but not stored.
 * @param buffer Buffer to store the line
 * @param max_len Maximum number of characters to read (including null terminator)
 * @return Number of characters read, or -1 on error
 */
int ext_io_getline(char *buffer, size_t max_len);

/**
 * Read an integer from input.
 * @param value Pointer to store the read integer
 * @return 0 on success, -1 on error
 */
int ext_io_scan_int(int *value);

/**
 * Read an unsigned integer from input.
 * @param value Pointer to store the read unsigned integer
 * @return 0 on success, -1 on error
 */
int ext_io_scan_uint(uint32_t *value);

/**
 * Check if input is available without blocking.
 * @return 1 if input is available, 0 if not, -1 on error
 */
int ext_io_kbhit(void);

/**
 * Read all pending input and discard it (non-blocking).
 * Useful for clearing the input buffer.
 */
void ext_io_clear_input(void);

/**
 * Set up non-blocking mode for input (Unix implementation).
 * @param enable 1 to enable non-blocking, 0 to disable
 */
void ext_io_set_nonblocking(int enable);

/**
 * Print an error message to stderr.
 * @param format printf-style format string
 * @param ... variable arguments
 */
int ext_io_eprintf(const char *format, ...);

/**
 * Exit the program with the given status code.
 * @param status Exit status code (0 for success, non-zero for failure)
 */
void ext_exit(int status);

/* ========== Random ========== */

/**
 * Initialize the randomness subsystem.
 * Should be called once before using other random functions.
 * @param seed Optional seed value for deterministic randomness (0 for non-deterministic)
 * @return 0 on success, -1 on error
 */
int ext_random_init(uint32_t seed);

/**
 * Fill a buffer with random bytes.
 * @param buffer Buffer to fill with random data
 * @param len Number of bytes to generate
 * @return 0 on success, -1 on error
 */
int ext_random_bytes(uint8_t *buffer, size_t len);

/**
 * Generate a random 32-bit unsigned integer.
 * @return Random 32-bit value
 */
uint32_t ext_random_u32(void);

/**
 * Generate a random value within a range [min, max].
 * @param min Minimum value (inclusive)
 * @param max Maximum value (inclusive)
 * @return Random value in range
 */
uint32_t ext_random_range(uint32_t min, uint32_t max);

/**
 * Generate a random 16-bit nonce.
 * @param nonce Buffer to store the nonce (must be at least 16 bytes)
 * @return 0 on success, -1 on error
 */
int ext_random_nonce(uint8_t *nonce);

/* ========== Utils ========== */

/**
 * Set a block of memory to a given value.
 * @param dest Pointer to the memory to fill
 * @param value Value to set (converted to unsigned char)
 * @param n Number of bytes to fill
 * @return Pointer to dest
 */
void *ext_memset(void *dest, int value, size_t n);

/**
 * Copy a block of memory.
 * @param dest Destination buffer
 * @param src Source buffer
 * @param n Number of bytes to copy
 * @return Pointer to dest
 */
void *ext_memcpy(void *dest, const void *src, size_t n);

/**
 * Compare two blocks of memory.
 * @param lhs First memory block
 * @param rhs Second memory block
 * @param n Number of bytes to compare
 * @return 0 if equal, negative if lhs < rhs, positive if lhs > rhs
 */
int ext_memcmp(const void *lhs, const void *rhs, size_t n);

#ifdef __cplusplus
}
#endif

#endif // EXT_SUPPORT_H_INCLUDED
