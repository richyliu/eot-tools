/**
 * ARM Cortex-M4 bare metal stub implementation of timer abstraction layer.
 * All functions are stubs that do nothing - actual implementations
 * should be provided by the application using hardware timers.
 */

#include "ext_timer.h"

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
