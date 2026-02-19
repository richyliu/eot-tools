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
void main_arm(void) {
#ifdef EOT_DEVICE
  eot_main();
#else
  hot_main();
#endif
}

void Reset_Handler(void) {
  main_arm();
  while (1) { }
}

void Default_Handler(void) {
  while (1) { }
}

__attribute__((section(".vectors")))
void (*const vector_table[])(void) = {
    (void (*)(void))0x20010000,
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
