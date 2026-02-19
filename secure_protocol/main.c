#include "devices.h"

#ifdef TARGET_UNIX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#ifdef TARGET_UNIX
int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s {eot|hot} [packet drop numbers...]\n", argv[0]);
    return EXIT_FAILURE;
  }

  if (argc > 2) {
    for (int i = 2; i < argc; i++) {
      int pkt_num = atoi(argv[i]);
      printf("Will drop packet %d\n", pkt_num);
      add_drop_packet(pkt_num);
    }
  }

  if (strcmp(argv[1], "eot") == 0) {
    return eot_main();
  } else if (strcmp(argv[1], "hot") == 0) {
    return hot_main();
  } else {
    fprintf(stderr, "Invalid argument: %s. Use 'eot' or 'hot'.\n", argv[1]);
    return EXIT_FAILURE;
  }

  return 0;
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
