#include "devices.h"
#include "ext_support.h"
#include "profiling.h"

typedef enum {
  MODE_DEFAULT = 0,
  MODE_TEST_PROFILE = 1,
  MODE_TEST_TIMING = 2
} protocol_mode_t;

void test_profile(void) {
  ext_io_puts("\n--- Starting Profile Test ---\n");
  PROFILE_START(test_profile_loop);
  for (volatile uint32_t i = 0; i < 100000; i++) {
    // Some dummy work to count instructions
  }
  PROFILE_END(test_profile_loop);
  ext_io_puts("--- Profile Test Complete ---\n");
}

void test_timing(void) {
  ext_io_puts("\n--- Starting Timing Test ---\n");
  ext_timer_t start, end;

  ext_io_puts("Testing 1s sleep...\n");
  ext_timer_now(&start);
  ext_timer_sleep_ms(1000);
  ext_timer_now(&end);
  ext_io_printf("Actual: %d ms\n", ext_timer_diff_ms(&end, &start));

  ext_io_puts("Testing 2s sleep...\n");
  ext_timer_now(&start);
  ext_timer_sleep_ms(2000);
  ext_timer_now(&end);
  ext_io_printf("Actual: %d ms\n", ext_timer_diff_ms(&end, &start));

  ext_io_puts("--- Timing Test Complete ---\n");
}

#ifdef TARGET_UNIX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#ifdef TARGET_UNIX
int main(int argc, char *argv[]) {
  protocol_mode_t mode = MODE_DEFAULT;
  const char *socket1 = NULL;
  const char *socket2 = NULL;
  int arg_start = 1;

  if (argc > 1) {
    if (strcmp(argv[1], "test_profile") == 0) {
      mode = MODE_TEST_PROFILE;
      arg_start = 2;
    } else if (strcmp(argv[1], "test_timing") == 0) {
      mode = MODE_TEST_TIMING;
      arg_start = 2;
    } else if (strcmp(argv[1], "default") == 0) {
      mode = MODE_DEFAULT;
      arg_start = 2;
    }
  }

  if (mode == MODE_DEFAULT) {
    if (argc < arg_start + 2) {
      fprintf(stderr,
              "Usage: %s [mode] <socket1> <socket2> [packet_drops...]\n",
              argv[0]);
      return 1;
    }
    socket1 = argv[arg_start];
    socket2 = argv[arg_start + 1];
    arg_start += 2;
    printf("[DEBUG] Using socket1: %s\n", socket1);
    printf("[DEBUG] Using socket2: %s\n", socket2);

    for (int i = arg_start; i < argc; i++) {
      int pkt_num = atoi(argv[i]);
      add_drop_packet(pkt_num);
    }
  }

  if (mode == MODE_TEST_PROFILE) {
    test_profile();
    return 0;
  } else if (mode == MODE_TEST_TIMING) {
    test_timing();
    return 0;
  }

#ifdef EOT_DEVICE
  return eot_main(socket1, socket2);
#else
  return hot_main(socket1, socket2);
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

void SysTick_Handler(void);
void Reset_Handler(void);
void Default_Handler(void);

void main_arm(void) {
  ext_io_init();
  ext_timer_init();
  ext_timer_init_cycles();

  int mode_val = 0;
  ext_io_puts("Select protocol mode:\n");
  ext_io_puts("  0: default\n");
  ext_io_puts("  1: test_profile\n");
  ext_io_puts("  2: test_timing\n");
  ext_io_puts("Mode: ");
  ext_io_flush();
  if (ext_io_scan_int(&mode_val) != 0) {
    mode_val = 0;
  }
  protocol_mode_t mode = (protocol_mode_t)mode_val;

  if (mode == MODE_DEFAULT) {
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
    eot_main(NULL, NULL);
#else
    hot_main(NULL, NULL);
#endif
  } else if (mode == MODE_TEST_PROFILE) {
    test_profile();
  } else if (mode == MODE_TEST_TIMING) {
    test_timing();
  }
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
  while (1) {
    asm volatile("wfi");
  }
}

void Default_Handler(void) {
  while (1) {
    asm volatile("wfi");
  }
}

__attribute__((section(".vectors"))) void (*const vector_table[])(void) = {
    (void (*)(void))&_estack, Reset_Handler,   Default_Handler, Default_Handler,
    Default_Handler,          Default_Handler, Default_Handler, Default_Handler,
    Default_Handler,          Default_Handler, Default_Handler, Default_Handler,
    Default_Handler,          Default_Handler, Default_Handler, SysTick_Handler,
};
#endif
