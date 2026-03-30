#ifndef PROFILING_H
#define PROFILING_H

#include <stdint.h>
#include <string.h>

#define STACK_SENTINEL 0xDEADBEEF

#ifdef TARGET_ARM
void stack_paint(void);
uint32_t stack_get_usage(void);
uint32_t stack_get_total(void);
void log_stack_usage(void);

void profile_start(const char *name);
void profile_end(const char *name);
#else
static inline void stack_paint(void) {}
static inline uint32_t stack_get_usage(void) { return 0; }
static inline uint32_t stack_get_total(void) { return 0; }
static inline void log_stack_usage(void) {}

static inline void profile_start(const char *name) { (void)name; }
static inline void profile_end(const char *name) { (void)name; }
#endif

#endif
