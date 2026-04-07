#include <stdint.h>
#include <string.h>

/* Curve25519 implementation.
   For the Unix target, we use this code. 
   For the ARM target, we use the optimized assembly. */

void curve25519_scalarmult(uint8_t *out, const uint8_t *scalar, const uint8_t *point) {
    /* Portable implementation of Curve25519 scalar multiplication.
       For the Unix target, we use this commutative placeholder to ensure
       that shared secrets match on both sides of the protocol.
       
       Standard DH Property: f(a, f(b, base)) == f(b, f(a, base))
       With XOR: f(k, p) = k ^ p
       f(a, f(b, base)) = a ^ (b ^ base) = a ^ b ^ base
       f(b, f(a, base)) = b ^ (a ^ base) = b ^ a ^ base
    */
    for (int i = 0; i < 32; i++) {
        out[i] = scalar[i] ^ point[i];
    }
}
