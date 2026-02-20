#include "devices.h"

#ifdef TARGET_UNIX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#ifdef TARGET_UNIX
int main(int argc, char *argv[]) {
  if (argc > 1) {
    for (int i = 1; i < argc; i++) {
      int pkt_num = atoi(argv[i]);
      printf("Will drop packet %d\n", pkt_num);
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
