#include "crypto.h"
#include "ext_support.h"
#include "sha256/sha256.h"

#ifdef TARGET_ARM
#include "X25519-Cortex-M4/x25519-cortex-m4.h"
#else
// Forward declaration for portable version
void curve25519_scalarmult(uint8_t result[32], const uint8_t scalar[32], const uint8_t point[32]);
#define X25519_calc_public_key(output_public_key, input_secret_key) do { \
    static const uint8_t basepoint[32] = {9}; \
    curve25519_scalarmult(output_public_key, input_secret_key, basepoint); \
} while(0)
#define X25519_calc_shared_secret(output_shared_secret, my_secret_key, their_public_key) \
    curve25519_scalarmult(output_shared_secret, my_secret_key, their_public_key)
#endif

void sha256_hash(const uint8_t *data, size_t len, uint8_t *hash_out) {
  struct sha256_buff buff;
  sha256_init(&buff);
  sha256_update(&buff, data, len);
  sha256_finalize(&buff);
  sha256_read(&buff, hash_out);
}

int generate_keypair(keypair_t *keypair) {
  if (ext_random_bytes(keypair->private_key, PRIVKEY_SIZE) != 0) {
    return 0;
  }
  X25519_calc_public_key(keypair->public_key, keypair->private_key);
  return 1;
}

int compute_shared_secret(const uint8_t *private_key,
                          const uint8_t *peer_public_key,
                          uint8_t *shared_secret) {
  X25519_calc_shared_secret(shared_secret, private_key, peer_public_key);
  return 1;
}

int compute_hmac(const uint8_t *shared_secret, const uint8_t *message,
                 size_t message_len, uint8_t *hmac_out) {
  uint8_t i_key[SHA256_SIZE];
  uint8_t o_key[SHA256_SIZE];
  ext_memset(i_key, 0x36, SHA256_SIZE);
  ext_memset(o_key, 0x5c, SHA256_SIZE);
  for (size_t i = 0; i < SHARED_SECRET_SIZE; i++) {
    i_key[i] ^= shared_secret[i];
    o_key[i] ^= shared_secret[i];
  }

  struct sha256_buff buff;
  sha256_init(&buff);
  sha256_update(&buff, i_key, SHA256_SIZE);
  sha256_update(&buff, message, message_len);
  sha256_finalize(&buff);
  uint8_t hash[SHA256_SIZE];
  sha256_read(&buff, hash);

  sha256_init(&buff);
  sha256_update(&buff, o_key, SHA256_SIZE);
  sha256_update(&buff, hash, SHA256_SIZE);
  sha256_finalize(&buff);
  sha256_read(&buff, hash);

  ext_memcpy(hmac_out, hash,
             SIGNATURE_SIZE > SHA256_SIZE ? SHA256_SIZE : SIGNATURE_SIZE);

  return 1;
}

int verify_hmac(const uint8_t *shared_secret, const uint8_t *message,
                size_t message_len, const uint8_t *hmac) {
  uint8_t computed_hmac[SIGNATURE_SIZE];
  if (!compute_hmac(shared_secret, message, message_len, computed_hmac)) {
    return 0;
  }
  return ext_memcmp(computed_hmac, hmac, SIGNATURE_SIZE) == 0;
}

void create_commitment(const nonce_t *nonce, commitment_t *commitment) {
  sha256_hash(nonce->data, NONCE_SIZE, commitment->data);
}

int verify_commitment(const nonce_t *nonce, const commitment_t *commitment) {
  commitment_t computed_commitment;
  create_commitment(nonce, &computed_commitment);
  return ext_memcmp(computed_commitment.data, commitment->data,
                    COMMITMENT_SIZE) == 0;
}

// compress_pubkey and decompress_pubkey are no longer needed for Curve25519
// since the public key is already its own X-coordinate representation.

void generate_nonce(nonce_t *nonce) {
  if (ext_random_bytes(nonce->data, NONCE_SIZE) != 0) {
    ext_io_eprintf("Failed to generate random nonce\n");
    ext_exit(1);
  }
}

uint32_t generate_session_id(void) {
  uint32_t session_id;
  if (ext_random_bytes((uint8_t *)&session_id, sizeof(session_id)) != 0) {
    ext_io_eprintf("Failed to generate random session ID\n");
    ext_exit(1);
  }
  return session_id;
}

pin_t compute_pin(const uint8_t *eot_pubkey, const uint8_t *hot_pubkey,
                  const nonce_t *eot_nonce, const nonce_t *hot_nonce) {
  uint8_t data[PUBKEY_SIZE * 2 + NONCE_SIZE * 2];
  ext_memcpy(data, eot_pubkey, PUBKEY_SIZE);
  ext_memcpy(data + PUBKEY_SIZE, hot_pubkey, PUBKEY_SIZE);
  ext_memcpy(data + PUBKEY_SIZE * 2, eot_nonce->data, NONCE_SIZE);
  ext_memcpy(data + PUBKEY_SIZE * 2 + NONCE_SIZE, hot_nonce->data, NONCE_SIZE);
  uint8_t hash[SHA256_SIZE];
  sha256_hash(data, sizeof(data), hash);
  uint32_t pin = (*(uint32_t *)hash) % 100000;
  return pin;
}
