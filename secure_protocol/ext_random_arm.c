/**
 * ARM Cortex-M4 bare metal stub implementation of randomness abstraction layer.
 * All functions are stubs that return fixed values - actual implementations
 * should use hardware RNG (TRNG) if available.
 */

#include "ext_random.h"
#include <stddef.h>

#define NONCE_SIZE 16

int ext_random_init(void) {
    return 0;
}

int ext_random_bytes(uint8_t *buffer, size_t len) {
    if (!buffer) {
        return -1;
    }
    for (size_t i = 0; i < len; i++) {
        buffer[i] = 0;
    }
    return 0;
}

uint32_t ext_random_u32(void) {
    return 0;
}

uint32_t ext_random_range(uint32_t min, uint32_t max) {
    if (min >= max) {
        return min;
    }
    return min;
}

int ext_random_nonce(uint8_t *nonce) {
    if (!nonce) {
        return -1;
    }
    return ext_random_bytes(nonce, NONCE_SIZE);
}
