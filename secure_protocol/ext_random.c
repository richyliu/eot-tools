/**
 * Unix implementation of randomness abstraction layer.
 * Uses /dev/urandom for random bytes.
 * This file can be replaced with a hardware RNG implementation.
 */

#include "ext_random.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

// NONCE_SIZE from devices.c
#define NONCE_SIZE 16

static int urandom_fd = -1;

int ext_random_init(void) {
    if (urandom_fd < 0) {
        urandom_fd = open("/dev/urandom", O_RDONLY);
        if (urandom_fd < 0) {
            return -1;
        }
    }
    return 0;
}

int ext_random_bytes(uint8_t *buffer, size_t len) {
    if (urandom_fd < 0) {
        if (ext_random_init() < 0) {
            return -1;
        }
    }
    
    size_t total_read = 0;
    while (total_read < len) {
        ssize_t n = read(urandom_fd, buffer + total_read, len - total_read);
        if (n < 0) {
            return -1;
        }
        total_read += (size_t)n;
    }
    return 0;
}

uint32_t ext_random_u32(void) {
    uint32_t val;
    ext_random_bytes((uint8_t *)&val, sizeof(val));
    return val;
}

uint32_t ext_random_range(uint32_t min, uint32_t max) {
    if (min >= max) {
        return min;
    }
    
    uint32_t range = max - min + 1;
    uint32_t limit = (UINT32_MAX / range) * range;
    uint32_t val;
    
    // Rejection sampling to avoid modulo bias
    do {
        val = ext_random_u32();
    } while (val >= limit);
    
    return min + (val % range);
}

int ext_random_nonce(uint8_t *nonce) {
    return ext_random_bytes(nonce, NONCE_SIZE);
}
