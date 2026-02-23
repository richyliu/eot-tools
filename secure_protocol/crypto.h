/**
 * Cryptographic functions for EOT/HOT protocol.
 * Includes ECC key generation, shared secret computation, and HMAC-SHA256
 * signing.
 */

#ifndef CRYPTO_H_INCLUDED
#define CRYPTO_H_INCLUDED

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ECC_CURVE uECC_secp256r1()
#define CURVE_SIZE 32
#define PUBKEY_SIZE (CURVE_SIZE * 2)
#define COMPRESSED_PUBKEY_SIZE (CURVE_SIZE + 1)
#define PRIVKEY_SIZE CURVE_SIZE
#define SHARED_SECRET_SIZE CURVE_SIZE
#define SIGNATURE_SIZE 6
#define NONCE_SIZE 16
#define SHA256_SIZE 32
#define COMMITMENT_SIZE 32

/**
 * ECC keypair structure containing private and public keys.
 */
typedef struct {
  uint8_t private_key[PRIVKEY_SIZE];
  uint8_t public_key[PUBKEY_SIZE];
} keypair_t;

/**
 * Commitment structure for nonce commitment scheme.
 */
typedef struct {
  uint8_t data[COMMITMENT_SIZE];
} commitment_t;

/**
 * Nonce structure for random nonces.
 */
typedef struct {
  uint8_t data[NONCE_SIZE];
} nonce_t;

/**
 * PIN type - 5-digit PIN derived from keys and nonces.
 */
typedef uint32_t pin_t;

/**
 * Compute SHA-256 hash of input data.
 *
 * @param data Pointer to the data to hash
 * @param len Length of the data in bytes
 * @param hash_out Buffer to store the hash output (32 bytes)
 */
void sha256_hash(const uint8_t *data, size_t len, uint8_t *hash_out);

/**
 * Generate a new ECC keypair using the secp256r1 curve.
 *
 * @param keypair Pointer to a keypair_t struct to hold the generated keys.
 * @return 1 on success, 0 on failure.
 */
int generate_keypair(keypair_t *keypair);

/**
 * Compute a shared secret using ECDH.
 *
 * @param private_key Pointer to the private key (32 bytes)
 * @param peer_public_key Pointer to the peer's public key (64 bytes)
 * @param shared_secret Buffer to store the shared secret (32 bytes)
 * @return 1 on success, 0 on failure.
 */
int compute_shared_secret(const uint8_t *private_key,
                          const uint8_t *peer_public_key,
                          uint8_t *shared_secret);

/**
 * Compute HMAC-SHA256 for a message using the shared secret.
 *
 * @param shared_secret Pointer to the shared secret (SHARED_SECRET_SIZE bytes).
 * @param message Pointer to the message to hash.
 * @param message_len Length of the message in bytes.
 * @param hmac_out Pointer to a buffer to hold the HMAC (SIGNATURE_SIZE bytes).
 * @return 1 on success, 0 on failure.
 */
int compute_hmac(const uint8_t *shared_secret, const uint8_t *message,
                 size_t message_len, uint8_t *hmac_out);

/**
 * Verify a HMAC-SHA256 for a message using the shared secret.
 *
 * @param shared_secret Pointer to the shared secret (SHARED_SECRET_SIZE bytes).
 * @param message Pointer to the message whose HMAC is to be verified.
 * @param message_len Length of the message in bytes.
 * @param hmac Pointer to the HMAC to verify (SIGNATURE_SIZE bytes).
 * @return 1 if the HMAC is valid, 0 otherwise.
 */
int verify_hmac(const uint8_t *shared_secret, const uint8_t *message,
                size_t message_len, const uint8_t *hmac);

/**
 * Create a commitment to a given nonce using SHA-256.
 *
 * @param nonce Pointer to the nonce_t struct containing the nonce.
 * @param commitment Pointer to the commitment_t struct to hold the resulting
 * commitment.
 */
void create_commitment(const nonce_t *nonce, commitment_t *commitment);

/**
 * Verify a commitment against a given nonce.
 *
 * @param nonce Pointer to the nonce_t struct containing the nonce.
 * @param commitment Pointer to the commitment_t struct containing the
 * commitment.
 * @return 1 if the commitment matches the nonce, 0 otherwise.
 */
int verify_commitment(const nonce_t *nonce, const commitment_t *commitment);

/**
 * Compress an ECC public key to compressed form.
 *
 * @param pubkey Pointer to the uncompressed public key (64 bytes)
 * @param compressed Buffer to store the compressed public key (33 bytes)
 */
void compress_pubkey(const uint8_t *pubkey, uint8_t *compressed);

/**
 * Decompress a compressed ECC public key.
 *
 * @param compressed Pointer to the compressed public key (33 bytes)
 * @param pubkey Buffer to store the uncompressed public key (64 bytes)
 * @return 1 on success, 0 if the key is invalid
 */
int decompress_pubkey(const uint8_t *compressed, uint8_t *pubkey);

/**
 * Generate a random nonce using the ext_random abstraction.
 *
 * @param nonce Pointer to the nonce_t struct to hold the generated nonce.
 */
void generate_nonce(nonce_t *nonce);

/**
 * Generate a random session ID using the ext_random abstraction.
 * @return Random session ID
 */
uint32_t generate_session_id(void);

/**
 * Computes a 5-digit PIN from the two public keys and nonces.
 * @param eot_pubkey EOT's public key
 * @param hot_pubkey HOT's public key
 * @param eot_nonce EOT's nonce
 * @param hot_nonce HOT's nonce
 * @return 5-digit PIN
 */
pin_t compute_pin(const uint8_t *eot_pubkey, const uint8_t *hot_pubkey,
                  const nonce_t *eot_nonce, const nonce_t *hot_nonce);

#ifdef __cplusplus
}
#endif

#endif
