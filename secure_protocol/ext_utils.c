/**
 * Implementation of utility functions.
 * Uses simple byte-by-byte implementations for portability.
 */

#include "ext_utils.h"

void *ext_memset(void *dest, int value, size_t n) {
    unsigned char *d = (unsigned char *)dest;
    unsigned char v = (unsigned char)value;
    while (n--) {
        *d++ = v;
    }
    return dest;
}

void *ext_memcpy(void *dest, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dest;
    const unsigned char *s = (const unsigned char *)src;
    while (n--) {
        *d++ = *s++;
    }
    return dest;
}

int ext_memcmp(const void *lhs, const void *rhs, size_t n) {
    const unsigned char *l = (const unsigned char *)lhs;
    const unsigned char *r = (const unsigned char *)rhs;
    while (n--) {
        if (*l != *r) {
            return *l - *r;
        }
        l++;
        r++;
    }
    return 0;
}
