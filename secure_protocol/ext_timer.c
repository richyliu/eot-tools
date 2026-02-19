/**
 * Unix implementation of time abstraction layer.
 * Uses clock_gettime for monotonic timing and usleep for delays.
 * This file can be replaced with a bare metal implementation.
 */

#include "ext_timer.h"
#include <time.h>
#include <unistd.h>

int ext_timer_init(void) {
    // Nothing to initialize for Unix implementation
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

void ext_timer_sleep_us(uint32_t us) {
    usleep(us);
}
