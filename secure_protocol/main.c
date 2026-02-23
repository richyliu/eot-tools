#include "devices.h"
#include "ext_support.h"
#include "profiling.h"

#ifdef TARGET_UNIX
#include <stdio.h>
#include <stdlib.h>
#endif

#ifdef TARGET_UNIX
int main(int argc, char *argv[]) {
  if (argc > 1) {
    for (int i = 1; i < argc; i++) {
      int pkt_num = atoi(argv[i]);
      add_drop_packet(pkt_num);
    }
  }

#ifdef EOT_DEVICE
  return eot_main();
#else
  return hot_main();
#endif
}
#endif

#ifdef TARGET_ARM
/* External symbols from linker script */
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sidata;
extern uint32_t _sbss;
extern uint32_t _ebss;
extern uint32_t _estack;

void main_arm(void) {
  ext_io_init();
  ext_timer_init();

  int seed;
  ext_io_puts("Seed for RNG:\n");
  ext_io_flush();
  ext_io_scan_int(&seed);
  ext_random_init(seed);

  while (1) {
    ext_io_puts("Enter packet number to drop (or -1 to stop):\n");
    ext_io_flush();
    int pkt_num;
    ext_io_scan_int(&pkt_num);
    if (pkt_num < 0) {
      break;
    }
    add_drop_packet(pkt_num);
  }

#ifdef EOT_DEVICE
  eot_main();
#else
  hot_main();
#endif
}

void Reset_Handler(void) {
  /* Copy .data from FLASH to RAM */
  uint32_t *src = &_sidata;
  uint32_t *dst = &_sdata;
  while (dst < &_edata) {
    *dst++ = *src++;
  }

  /* Zero .bss */
  dst = &_sbss;
  while (dst < &_ebss) {
    *dst++ = 0;
  }

  /* Paint stack for profiling */
  stack_paint();

  /* Call main */
  main_arm();

  /* Should never return, but if it does, loop forever */
  while (1) { }
}

void Default_Handler(void) {
  while (1) { }
}

__attribute__((section(".vectors")))
void (*const vector_table[])(void) = {
    (void (*)(void))&_estack,
    Reset_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
};
#endif
