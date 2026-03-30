#include "profiling.h"
#include "ext_support.h"

#ifdef TARGET_ARM
#include "qemu_tcg_plugins/inscount.h"
extern uint32_t _ebss;
extern uint32_t _estack;

void stack_paint(void) {
  uint32_t *ptr = &_ebss;
  while (ptr < &_estack) {
    *ptr++ = STACK_SENTINEL;
  }
}

uint32_t stack_get_usage(void) {
  uint32_t *ptr = &_ebss;
  while (ptr < &_estack && *ptr == STACK_SENTINEL) {
    ptr++;
  }
  return (uint32_t)((char *)&_estack - (char *)ptr);
}

uint32_t stack_get_total(void) {
  return (uint32_t)((char *)&_estack - (char *)&_ebss);
}

void log_stack_usage(void) {
  uint32_t used = stack_get_usage();
  uint32_t total = stack_get_total();
  if (total == 0)
    return;
  ext_io_printf("Stack: %u/%u bytes (%u%%)\n", used, total,
                (used * 100) / total);
}

void profile_start(const char *name) {
  strlcpy((char *)PROFILE_MAGIC_NAME_START, name, PROFILE_MAGIC_NAME_SIZE);
  *(volatile int *)(PROFILE_MAGIC_CONTROL) = 1;
}

void profile_end(const char *name) {
  (void)name;
  *(volatile int *)(PROFILE_MAGIC_CONTROL) = 2;
}
#endif
