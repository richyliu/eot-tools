/**
 * Utility functions abstraction layer.
 * Provides portable implementations of common memory and string functions.
 * Can be swapped out for bare metal implementations at build time.
 */

#ifndef EXT_UTILS_H_INCLUDED
#define EXT_UTILS_H_INCLUDED

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set a block of memory to a given value.
 * @param dest Pointer to the memory to fill
 * @param value Value to set (converted to unsigned char)
 * @param n Number of bytes to fill
 * @return Pointer to dest
 */
void *ext_memset(void *dest, int value, size_t n);

/**
 * Copy a block of memory.
 * @param dest Destination buffer
 * @param src Source buffer
 * @param n Number of bytes to copy
 * @return Pointer to dest
 */
void *ext_memcpy(void *dest, const void *src, size_t n);

/**
 * Compare two blocks of memory.
 * @param lhs First memory block
 * @param rhs Second memory block
 * @param n Number of bytes to compare
 * @return 0 if equal, negative if lhs < rhs, positive if lhs > rhs
 */
int ext_memcmp(const void *lhs, const void *rhs, size_t n);

#ifdef __cplusplus
}
#endif

#endif // EXT_UTILS_H_INCLUDED
