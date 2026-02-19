/**
 * Randomness abstraction layer for the EOT/HOT protocol.
 * Provides portable randomness that can be swapped out for
 * hardware RNG implementations at build time.
 */

#ifndef EXT_RANDOM_H_INCLUDED
#define EXT_RANDOM_H_INCLUDED

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the randomness subsystem.
 * Should be called once before using other random functions.
 * @return 0 on success, -1 on error
 */
int ext_random_init(void);

/**
 * Fill a buffer with random bytes.
 * @param buffer Buffer to fill with random data
 * @param len Number of bytes to generate
 * @return 0 on success, -1 on error
 */
int ext_random_bytes(uint8_t *buffer, size_t len);

/**
 * Generate a random 32-bit unsigned integer.
 * @return Random 32-bit value
 */
uint32_t ext_random_u32(void);

/**
 * Generate a random value within a range [min, max].
 * @param min Minimum value (inclusive)
 * @param max Maximum value (inclusive)
 * @return Random value in range
 */
uint32_t ext_random_range(uint32_t min, uint32_t max);

/**
 * Generate a random 16-bit nonce.
 * @param nonce Buffer to store the nonce (must be at least 16 bytes)
 * @return 0 on success, -1 on error
 */
int ext_random_nonce(uint8_t *nonce);

#ifdef __cplusplus
}
#endif

#endif // EXT_RANDOM_H_INCLUDED
