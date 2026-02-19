/**
 * Time abstraction layer for the EOT/HOT protocol.
 * Provides a portable time interface that can be swapped out for
 * bare metal implementations at build time.
 */

#ifndef EXT_TIMER_H_INCLUDED
#define EXT_TIMER_H_INCLUDED

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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

/**
 * Sleep for a specified number of microseconds.
 * @param us Number of microseconds to sleep
 */
void ext_timer_sleep_us(uint32_t us);

#ifdef __cplusplus
}
#endif

#endif // EXT_TIMER_H_INCLUDED
