#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "devices.h"


int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s {eot|hot} [packet drop numbers...]\n", argv[0]);
    return EXIT_FAILURE;
  }

  if (argc > 2) {
    // use subsequent arguments as packet drop numbers (for testing)
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
