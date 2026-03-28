/**
 * ARM Cortex-M4 bare metal support functions.
 * Provides minimal implementations of standard library functions
 * required by micro-ecc and sha256 libraries.
 */

#include <stddef.h>

void *memset(void *dest, int value, size_t n) {
  unsigned char *d = (unsigned char *)dest;
  unsigned char v = (unsigned char)value;
  while (n--) {
    *d++ = v;
  }
  return dest;
}

void *memcpy(void *dest, const void *src, size_t n) {
  unsigned char *d = (unsigned char *)dest;
  const unsigned char *s = (const unsigned char *)src;
  while (n--) {
    *d++ = *s++;
  }
  return dest;
}

size_t strlcpy(char *dst, const char *src, size_t size) {
  size_t len = 0;
  while (len < size - 1 && src[len] != '\0') {
    dst[len] = src[len];
    len++;
  }
  if (size > 0) {
    dst[len] = '\0';
  }
  return len;
}