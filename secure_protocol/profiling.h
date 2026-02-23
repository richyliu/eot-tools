#ifndef PROFILING_H
#define PROFILING_H

#include <stdint.h>

#define STACK_SENTINEL 0xDEADBEEF

#ifdef TARGET_ARM
extern uint32_t _ebss;
extern uint32_t _estack;

static inline void stack_paint(void) {
    uint32_t *ptr = &_ebss;
    while (ptr < &_estack) {
        *ptr++ = STACK_SENTINEL;
    }
}

static inline uint32_t stack_get_usage(void) {
    uint32_t *ptr = &_ebss;
    while (ptr < &_estack && *ptr == STACK_SENTINEL) {
        ptr++;
    }
    return (uint32_t)((char *)&_estack - (char *)ptr);
}

static inline uint32_t stack_get_total(void) {
    return (uint32_t)((char *)&_estack - (char *)&_ebss);
}

void log_stack_usage(void);

#else
static inline void stack_paint(void) {}
static inline uint32_t stack_get_usage(void) { return 0; }
static inline uint32_t stack_get_total(void) { return 0; }
static inline void log_stack_usage(void) {}
#endif

#endif
