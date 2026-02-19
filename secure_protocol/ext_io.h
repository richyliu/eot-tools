/**
 * IO abstraction layer for the EOT/HOT protocol.
 * Provides portable input/output using nanoprintf instead of printf.
 * Can be swapped out for bare metal UART implementations at build time.
 */

#ifndef EXT_IO_H_INCLUDED
#define EXT_IO_H_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

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
int ext_io_scan_uint(unsigned int *value);

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

#ifdef __cplusplus
}
#endif

#endif // EXT_IO_H_INCLUDED
