/**
 * This is an upgraded protocol for communication between end-of-train
 * devices (EOTD) and head-of-train devices (HOTD) that supports
 * message authentication using ECDH and HMAC-SHA256, as well as
 * a legacy mode for backward compatibility.
 *
 * To ensure minimal operator changes, the pairing process is designed
 * to be as similar to the old process as possible. The old pairing
 * procedure is described below:
 * 1. Railman installs the EOTD to the end of the train and relays to
 *    the engineer at the head of the train the EOT ID.
 * 2. The engineer inputs the 5-digit EOT ID into the HOTD.
 * 3. The railman pushes the TEST button on the EOTD, which broadcasts
 *    an ARM message.
 * 4. The engineer at the head of the train has 5 seconds to press the
 *    ARM button to confirm pairing.
 * Sources:
 * - https://www.youtube.com/watch?v=UI4a9ygz_pI&t=316
 * - https://vimeo.com/groups/310557/videos/124589083
 */

#include "ext_utils.h"

#include "micro-ecc/uECC.h"
#include "sha256/sha256.h"

#include "devices.h"
#include "comm.h"
#include "ext_timer.h"
#include "ext_io.h"
#include "ext_random.h"

// for testing packet dropping
static int pkt_dropped[10];
#define PACKET_SEND_DELAY_MS_PER_BYTE 15 // simulate a delay in sending packets

#define DEFAULT_TIMEOUT_MS 300
#define HOT_ADV_INTERVAL_MS 1000
#define PAIRING_TIMEOUT 6000
#define HOT_RETRANSMIT_INTERVAL_MS 2000
#define EOT_WAIT_ADV_TIMEOUT_MS 30000

#define ECC_CURVE uECC_secp256r1()
// all sizes are in bytes
#define CURVE_SIZE 32
#define PUBKEY_SIZE (CURVE_SIZE * 2)
#define COMPRESSED_PUBKEY_SIZE (CURVE_SIZE + 1)
#define PRIVKEY_SIZE CURVE_SIZE
#define SHARED_SECRET_SIZE CURVE_SIZE
#define SIGNATURE_SIZE 6 // number of bytes to use in HMAC-SHA256 signature

#define NONCE_SIZE 16
#define SHA256_SIZE 32
#define COMMITMENT_SIZE 32 // SHA-256 output size

#define MAX_MSG_LEN 256
#define MAX_PKT_LEN 512

typedef uint32_t session_id_t;

enum eot_msg {
  // these are messages sent from EOT to HOT
  EOT_MSG_PUBKEY = 0,
  EOT_MSG_NONCE,
  EOT_MSG_STATUS,    // status update response
  EOT_MSG_EMERGENCY, // emergency brake confirmation
  EOT_MSG_UPGRADE,   // sent in legacy mdoe to request protocol upgrade
};
enum hot_msg {
  // these are messages sent from HOT to EOT
  HOT_MSG_ADV = 0,
  HOT_MSG_PUBKEY_AND_COMMIT,
  HOT_MSG_NONCE,
  HOT_MSG_STATUS,    // status update request
  HOT_MSG_EMERGENCY, // emergency brake request
  HOT_MSG_DISCONNECT
} hot;
typedef uint8_t msg_type_t;

enum eot_state {
  EOT_IDLE = 0,    // initial state
  EOT_WAIT_ADV,    // waiting for advertisement from HOT
  EOT_KEY_EX_1,    // sent our pubkey, waiting for theirs
  EOT_KEY_EX_2,    // received their pubkey and commitment, sending our nonce and waiting for theirs
  EOT_PAIRED,
  EOT_LEGACY,
};

enum hot_state {
  HOT_IDLE = 0,     // initial state
  HOT_ADV,          // sending periodic advertisements
  HOT_KEY_EX_1,     // received their pubkey, sending our pubkey and commitment, waiting for their nonce
  HOT_WAIT_FOR_PIN, // received their nonce, waiting for user to input PIN
  HOT_PAIRED,
  HOT_WAIT_FOR_STATUS,
  HOT_WAIT_FOR_EMERGENCY, // wait for confirmation of emergency brake
  HOT_LEGACY,
  HOT_LEGACY_ARMED,
};

typedef struct {
  uint8_t batt_cond;                 // 2 bits
  uint16_t pressure;                 // 7 bits
  uint16_t batt_charge_used;         // 7 bits
  uint8_t valve_circuit_operational; // 1 bit
  uint8_t confirmation_indicator;    // 1 bit
  uint8_t turbine_status;            // 1 bit
  uint8_t motion_detection;          // 1 bit
  uint8_t marker_light_battery_weak; // 1 bit
  uint8_t marker_light_status;       // 1 bit
} eot_status_t;

typedef struct {
  uint8_t private[PRIVKEY_SIZE];
  uint8_t public[PUBKEY_SIZE];
} keypair_t;

typedef struct {
  uint8_t data[COMMITMENT_SIZE];
} commitment_t;

typedef struct {
  uint8_t data[NONCE_SIZE];
} nonce_t;

typedef uint32_t pin_t; // 5-digit PIN
typedef uint32_t msg_ctr_t; // used for replay protection
typedef uint32_t unit_id_t;

typedef struct {
  uint32_t timeout_ms;
  comm_handle_t *comm_h;
} communicator_t;

typedef struct {
  session_id_t session_id;
  keypair_t eot_keys;
  keypair_t hot_keys;
  uint8_t shared_secret[SHARED_SECRET_SIZE];
  // eot_commitment not needed, as HOT is the first sender
  commitment_t hot_commitment;
  nonce_t eot_nonce;
  nonce_t hot_nonce;
  pin_t pin;
  msg_ctr_t ctr; // message counter for replay protection
} conn_info_t;

// Use ext_timer_t from ext_timer.h
typedef ext_timer_t protocol_timer_t;

static inline void timer_now(protocol_timer_t *t) {
  ext_timer_now(t);
}

static inline int timer_diff_ms(const protocol_timer_t *end, const protocol_timer_t *start) {
  return ext_timer_diff_ms(end, start);
}


void sha256(const uint8_t *data, size_t len, uint8_t *hash_out) {
  struct sha256_buff buff;
  sha256_init(&buff);
  sha256_update(&buff, data, len);
  sha256_finalize(&buff);
  sha256_read(&buff, hash_out);
}

/**
 * Generate a new ECC keypair using the secp256r1 curve.
 * 
 * @param keypair Pointer to a keypair_t struct to hold the generated keys.
 * @return 1 on success, 0 on failure.
 */
int generate_keypair(keypair_t *keypair) {
  return uECC_make_key(keypair->public, keypair->private, ECC_CURVE);
}

int compute_shared_secret(const uint8_t *private_key, const uint8_t *peer_public_key, uint8_t *shared_secret) {
  int ret = uECC_shared_secret(peer_public_key, private_key, shared_secret, ECC_CURVE);
  return ret;
}

/**
 * Generate HMAC-SHA256 signature for a message using ECDSA with the given private key.
 *
 * @param shared_secret Pointer to the shared secret (SHARED_SECRET_SIZE bytes).
 * @param message Pointer to the message to sign.
 * @param message_len Length of the message in bytes.
 * @param signature Pointer to a buffer to hold the signature (SIGNATURE_SIZE bytes).
 * @return 1 on success, 0 on failure.
 */
int sign_message(const uint8_t *shared_secret, const uint8_t *message, size_t message_len, uint8_t *signature) {
  uint8_t i_key[SHA256_SIZE];
  uint8_t o_key[SHA256_SIZE];
  // Prepare inner and outer keys for HMAC
  ext_memset(i_key, 0x36, SHA256_SIZE);
  ext_memset(o_key, 0x5c, SHA256_SIZE);
  for (size_t i = 0; i < SHARED_SECRET_SIZE; i++) {
    i_key[i] ^= shared_secret[i];
    o_key[i] ^= shared_secret[i];
  }

  // Compute HMAC-SHA256
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

  ext_memcpy(signature, hash, SIGNATURE_SIZE > SHA256_SIZE ? SHA256_SIZE : SIGNATURE_SIZE);

  return 1;
}

/**
 * Verify a HMAC-SHA256 signature for a message using ECDSA with the given shared secret.
 *
 * @param shared_secret Pointer to the shared secret (SHARED_SECRET_SIZE bytes).
 * @param message Pointer to the message whose signature is to be verified.
 * @param message_len Length of the message in bytes.
 * @param signature Pointer to the signature to verify (SIGNATURE_SIZE bytes).
 * @return 1 if the signature is valid, 0 otherwise.
 */
int verify_signature(const uint8_t *shared_secret, const uint8_t *message, size_t message_len, const uint8_t *signature) {
  uint8_t computed_signature[SIGNATURE_SIZE];
  if (!sign_message(shared_secret, message, message_len, computed_signature)) {
    return 0;
  }
  return ext_memcmp(computed_signature, signature, SIGNATURE_SIZE) == 0;
}

/**
 * Create a commitment to a given nonce using SHA-256.
 * 
 * @param nonce Pointer to the nonce_t struct containing the nonce.
 * @param commitment Pointer to the commitment_t struct to hold the resulting commitment.
 */
void create_commitment(const nonce_t *nonce, commitment_t *commitment) {
  sha256(nonce->data, NONCE_SIZE, commitment->data);
}

/**
 * Verify a commitment against a given nonce.
 *
 * @param nonce Pointer to the nonce_t struct containing the nonce.
 * @param commitment Pointer to the commitment_t struct containing the commitment.
 * @return 1 if the commitment matches the nonce, 0 otherwise.
 */
int verify_commitment(const nonce_t *nonce, const commitment_t *commitment) {
  commitment_t computed_commitment;
  create_commitment(nonce, &computed_commitment);
  return ext_memcmp(computed_commitment.data, commitment->data, COMMITMENT_SIZE) == 0;
}

void compress_pubkey(const uint8_t *pubkey, uint8_t *compressed) {
  uECC_compress(pubkey, compressed, ECC_CURVE);
}

int decompress_pubkey(const uint8_t *compressed, uint8_t *pubkey) {
  uECC_decompress(compressed, pubkey, ECC_CURVE);
  if (!uECC_valid_public_key(pubkey, ECC_CURVE)) {
    ext_io_printf("Invalid public key after decompression\n");
    return 0;
  }
  return 1;
}

/**
 * Generate a random nonce using the ext_random abstraction.
 * 
 * @param nonce Pointer to the nonce_t struct to hold the generated nonce.
 */
void generate_nonce(nonce_t *nonce) {
  if (ext_random_bytes(nonce->data, NONCE_SIZE) != 0) {
    ext_io_eprintf("Failed to generate random nonce\n");
    ext_exit(1);
  }
}

/**
 * Generate a random session ID using the ext_random abstraction.
 * @return Random session ID
 */
session_id_t generate_session_id() {
  session_id_t session_id;
  if (ext_random_bytes((uint8_t *)&session_id, sizeof(session_id)) != 0) {
    ext_io_eprintf("Failed to generate random session ID\n");
    ext_exit(1);
  }
  return session_id;
}

/**
 * Computes a 5-digit PIN from the two public keys and nonces
 * @param eot_pubkey EOT's public key
 * @param hot_pubkey HOT's public key
 * @param eot_nonce EOT's nonce
 * @param hot_nonce HOT's nonce
 * @return 5-digit PIN
 */
pin_t compute_pin(const uint8_t *eot_pubkey, const uint8_t *hot_pubkey,
                  const nonce_t *eot_nonce, const nonce_t *hot_nonce) {
  uint8_t data[PUBKEY_SIZE * 2 + NONCE_SIZE * 2];
  ext_memcpy(data, eot_pubkey, PUBKEY_SIZE);
  ext_memcpy(data + PUBKEY_SIZE, hot_pubkey, PUBKEY_SIZE);
  ext_memcpy(data + PUBKEY_SIZE * 2, eot_nonce->data, NONCE_SIZE);
  ext_memcpy(data + PUBKEY_SIZE * 2 + NONCE_SIZE, hot_nonce->data, NONCE_SIZE);
  uint8_t hash[SHA256_SIZE];
  sha256(data, sizeof(data), hash);
  // take first 4 bytes of hash, convert to integer, mod 100000 to get 5-digit PIN
  uint32_t pin = (*(uint32_t*)hash) % 100000;
  return pin;
}


// pass in NULL for shared_secret for no signature
void comm_send(communicator_t* comm,
               const session_id_t session_id,
               const msg_type_t msg_type,
               const uint8_t *msg,
               const size_t msg_len,
               const uint8_t *shared_secret) {
  static int pkt_ctr = 0;
  pkt_ctr++;
  for (int i = 0; i < sizeof(pkt_dropped)/sizeof(pkt_dropped[0]); i++) {
    if (pkt_dropped[i] == pkt_ctr) {
      ext_io_printf("[WARN] dropping packet %d for testing\n", pkt_ctr);
      return;
    }
  }

  size_t total_len = sizeof(session_id_t) + sizeof(msg_type_t) + msg_len;
  if (total_len > MAX_PKT_LEN) {
    ext_io_eprintf("Message too long to send (%zu bytes)\n", total_len);
    ext_exit(1);
  }
  uint8_t buffer[MAX_PKT_LEN];

  ext_memcpy(buffer, &session_id, sizeof(session_id_t));
  ext_memcpy(buffer + sizeof(session_id_t), &msg_type, sizeof(msg_type_t));
  ext_memcpy(buffer + sizeof(session_id_t) + sizeof(msg_type_t), msg, msg_len);

  if (shared_secret != NULL) {
    uint8_t signature[SIGNATURE_SIZE];
    if (!sign_message(shared_secret, buffer, total_len, signature)) {
      ext_io_eprintf("Failed to sign message\n");
      ext_exit(1);
    }
    // append signature to the end of the message
    if (total_len + SIGNATURE_SIZE > MAX_PKT_LEN) {
      ext_io_eprintf("Message too long to send with signature (%zu bytes)\n", total_len + SIGNATURE_SIZE);
      ext_exit(1);
    }
    ext_memcpy(buffer + total_len, signature, SIGNATURE_SIZE);
    total_len += SIGNATURE_SIZE;
  }

  ext_io_printf("[INFO] sending message of length %zu (session_id=%u, msg_type=%d)", total_len, session_id, msg_type);
  ext_io_flush();
  // simulate a packet delay in sending
  for (int i = 0; i < total_len / 5; i++) {
    ext_timer_sleep_us(5 * PACKET_SEND_DELAY_MS_PER_BYTE * 1000);
    ext_io_putc('.');
    ext_io_flush();
  }
  ext_io_puts(" sent\n");

  if (comm_send_raw(comm->comm_h, buffer, total_len) != (ssize_t)total_len) {
    ext_io_eprintf("Failed to send message\n");
    ext_exit(1);
  }
}

void comm_send_legacy(communicator_t* comm,
                      const unit_id_t unit_id,
                      const uint8_t *msg,
                      const size_t msg_len) {
  uint32_t legacy_header;
  ext_memcpy(&legacy_header, "OLD!", 4);
  size_t total_len = sizeof(legacy_header) + sizeof(unit_id_t) + msg_len;
  if (total_len > MAX_PKT_LEN) {
    ext_io_eprintf("Message too long to send (%zu bytes)\n");
    ext_exit(1);
  }
  uint8_t buffer[MAX_PKT_LEN];
  ext_memcpy(buffer, &legacy_header, sizeof(legacy_header));
  ext_memcpy(buffer + sizeof(legacy_header), &unit_id, sizeof(unit_id_t));
  ext_memcpy(buffer + sizeof(legacy_header) + sizeof(unit_id_t), msg, msg_len);
  if (comm_send_raw(comm->comm_h, buffer, total_len) != (ssize_t)total_len) {
    ext_io_eprintf("Failed to send legacy message\n");
    ext_exit(1);
  }
  ext_io_printf("sent legacy message of length %zu\n", total_len);
}

// returns -1 on timeout, -2 on signature failure, or length of received message
// only checks for signature if shared_secret is not NULL
// returns negative length of message on legacy message
ssize_t comm_recv(communicator_t* comm,
                  session_id_t *session_id,
                  msg_type_t *msg_type,
                  uint8_t *msg,
                  const size_t max_msg_len,
                  const uint8_t *shared_secret) {
  uint8_t buffer[MAX_PKT_LEN];
  size_t header_len = sizeof(session_id_t) + sizeof(msg_type_t);

  ssize_t recv_len = comm_recv_raw(comm->comm_h, buffer, sizeof(buffer));
  if (recv_len == -1) {
    // timeout
    return -1;
  }
  if (recv_len == -2) {
    ext_io_eprintf("IPC receive error\n");
    ext_exit(1);
  }

  if (recv_len >= 4 && ext_memcmp(buffer, "OLD!", 4) == 0) {
    // legacy message must include the 4-byte header and unit_id, although we keep unit_id as part of msg
    if (recv_len < 4 + sizeof(unit_id_t)) {
      ext_io_eprintf("Received legacy message too short (%zd bytes)\n", recv_len);
      return -2;
    }
    size_t payload_len = recv_len - 4;
    if (payload_len > max_msg_len) {
      ext_io_eprintf("Received legacy message too long (%zu bytes)\n", payload_len);
      ext_exit(1);
    }
    ext_memcpy(msg, buffer + 4, payload_len);
    return -((ssize_t)payload_len); // negative length indicates legacy message
  }

  size_t payload_len = recv_len - header_len;

  if (shared_secret != NULL) {
    // verify signature
    if (recv_len < header_len + SIGNATURE_SIZE) {
      ext_io_eprintf("Received message too short for signature (data: %zd bytes)\n", recv_len);
      return -2;
    }
    uint8_t signature[SIGNATURE_SIZE];
    ext_memcpy(signature, buffer + recv_len - SIGNATURE_SIZE, SIGNATURE_SIZE);
    if (!verify_signature(shared_secret, buffer, recv_len - SIGNATURE_SIZE, signature)) {
      ext_io_eprintf("Signature verification failed\n");
      return -2;
    }
    payload_len -= SIGNATURE_SIZE;
  }

  if (payload_len > max_msg_len) {
    ext_io_eprintf("Received message too long (data: %zu bytes, total: %zd bytes)\n", payload_len, recv_len);
    ext_exit(1);
  }

  ext_memcpy(session_id, buffer, sizeof(session_id_t));
  ext_memcpy(msg_type, buffer + sizeof(session_id_t), sizeof(msg_type_t));
  ext_memcpy(msg, buffer + header_len, payload_len);

  return payload_len;
}

/**
 * Get current EOT data to send. Not implemented, returns dummy data.
 */
void get_eot_status(eot_status_t *status) {
  // In a real implementation, gather actual data from sensors
  status->batt_cond = 2;                 // Good
  status->pressure = 100;                 // Example pressure
  status->batt_charge_used = 50;          // Example charge used
  status->valve_circuit_operational = 1;  // Operational
  status->confirmation_indicator = 0;     // Off
  status->turbine_status = 1;             // Spinning
  status->motion_detection = 0;           // No motion
  status->marker_light_battery_weak = 0;  // Battery good
  status->marker_light_status = 1;        // On
}

void display_eot_status(eot_status_t *status) {
  // Simplified display without using time functions
  ext_io_puts("[EOT Status]:\n");
  ext_io_printf("  Battery Condition: %u\n", status->batt_cond);
  ext_io_printf("  Pressure: %u\n", status->pressure);
  ext_io_printf("  Battery Charge Used: %u\n", status->batt_charge_used);
  ext_io_printf("  Valve Circuit Operational: %u\n", status->valve_circuit_operational);
  ext_io_printf("  Confirmation Indicator: %u\n", status->confirmation_indicator);
  ext_io_printf("  Turbine Status: %u\n", status->turbine_status);
  ext_io_printf("  Motion Detection: %u\n", status->motion_detection);
  ext_io_printf("  Marker Light Battery Weak: %u\n", status->marker_light_battery_weak);
  ext_io_printf("  Marker Light Status: %u\n", status->marker_light_status);
  ext_io_puts("\n");
}

void eot_emergency_brake() {
  // In a real implementation, trigger the emergency brake mechanism
  ext_io_puts("EOT: Emergency brake activated!\n");
}

/**
 * Simulate waiting for the user to press the ARM button.
 * Uses STDIN for simulation.
 */
void wait_for_arm_button_press() {
  // clear any existing input
  ext_io_clear_input();
  
  ext_io_puts("Press enter to simulate ARM button press...\n");
  ext_io_getc();
}

void eot_handle_legacy_message(const uint8_t *msg, size_t msg_len) {
  // In a real implementation, handle legacy messages appropriately
  ext_io_puts("EOT: Received legacy message (not implemented)\n");
}

void hot_handle_legacy_message(const uint8_t *msg, size_t msg_len) {
  // In a real implementation, handle legacy messages appropriately
  ext_io_puts("HOT: Received legacy message (not implemented)\n");
}

void eot_run(communicator_t *comm, unit_id_t unit_id) {
  // EOT main loop

  conn_info_t conn; // current connection info
  ext_memset(&conn, 0, sizeof(conn));
  session_id_t recved_session_id = 0;
  enum eot_state state = EOT_IDLE;
  uint8_t msg[MAX_MSG_LEN];
  msg_type_t msg_type;
  eot_status_t status;
  protocol_timer_t adv_start;
  protocol_timer_t pairing_start;
  protocol_timer_t now;
  timer_now(&adv_start);
  timer_now(&pairing_start);
  timer_now(&now);
  uint8_t* shared_secret;
  int choice = 0;

  while (1) {
    // use shared_secret once paired
    if (state == EOT_PAIRED) {
      shared_secret = conn.shared_secret;
    } else {
      shared_secret = NULL;
    }
    ssize_t recv_len = comm_recv(comm, &recved_session_id, &msg_type, msg, sizeof(msg), shared_secret);
    timer_now(&now);
    if (recv_len == -1) {
      // timeout from recv, no message received
      switch (state) {
      case EOT_IDLE:
        ext_io_puts("EOT_IDLE: waiting for user to push TEST button\n");
        ext_io_puts("Options:\n");
        ext_io_puts("  1: Push TEST button to start pairing\n");
        ext_io_printf("  2: Enter legacy mode (hold TEST button for 5 seconds) (ID: %05u)\n", unit_id);
        ext_io_puts("Enter choice: ");
        ext_io_flush();
        ext_io_scan_int(&choice);
        if (choice == 1) {
          ext_io_puts("Button pressed, waiting for HOT advertisement...\n");
          timer_now(&adv_start);
          state = EOT_WAIT_ADV;
        } else if (choice == 2) {
          ext_io_puts("Entering legacy mode and sending legacy ARM command\n");
          state = EOT_LEGACY;
          // also send protocol upgrade request to prevent pairing in
          // legacy mode if both devices support the new protocol
          comm_send(comm, 0, (msg_type_t)EOT_MSG_UPGRADE, (uint8_t*)&unit_id, sizeof(unit_id), NULL);
          comm_send_legacy(comm, unit_id, (uint8_t*)"ARM", 3);
          ext_io_puts("Press enter to exit legacy mode back to idle state...\n");
        } else {
          ext_io_puts("Invalid choice, staying in idle state.\n");
        }
        break;
      case EOT_WAIT_ADV:
        if (timer_diff_ms(&now, &adv_start) >= EOT_WAIT_ADV_TIMEOUT_MS) {
          ext_io_puts("EOT: Waiting for HOT timed out. Returning to idle state.\n");
          state = EOT_IDLE;
          ext_memset(&conn, 0, sizeof(conn));
        }
        break;
      case EOT_KEY_EX_1:
      case EOT_KEY_EX_2:
        if (timer_diff_ms(&now, &pairing_start) >= PAIRING_TIMEOUT) {
          ext_io_puts("EOT: Pairing timed out. Returning to idle state.\n");
          state = EOT_IDLE;
          ext_memset(&conn, 0, sizeof(conn));
        }
        break;
      case EOT_PAIRED:
        // TODO: periodically send status updates
        // check for user input (non-blocking) to disconnect
        {
          ext_io_set_nonblocking(1);
          char buf[16];
          if (ext_io_getline(buf, sizeof(buf)) >= 0) {
            ext_io_puts("Pressed enter (ARM button), disconnecting and searching for new HOT...\n");
            timer_now(&adv_start);
            state = EOT_WAIT_ADV;
            ext_memset(&conn, 0, sizeof(conn));
          }
          ext_io_set_nonblocking(0);
        }
        break;
      case EOT_LEGACY:
        // check for user input (non-blocking) to exit legacy mode
        {
          ext_io_set_nonblocking(1);
          char buf[16];
          if (ext_io_getline(buf, sizeof(buf)) >= 0) {
            ext_io_puts("Pressed enter (TEST button), exiting legacy mode and returning to idle state...\n");
            state = EOT_IDLE;
          }
          ext_io_set_nonblocking(0);
        }
        break;
      default:
        break;
      }
    } else if (recv_len == -2) {
      // signature verification failed
      ext_io_puts("EOT: received message with invalid signature, ignoring.\n");
    } else if (recv_len < 0) {
      // legacy message received
      if (state != EOT_LEGACY) {
        ext_io_puts("EOT: received legacy message while not in legacy state, ignoring.\n");
        continue;
      }
      if (-recv_len < (ssize_t)sizeof(unit_id_t)) {
        ext_io_eprintf("EOT: received legacy message too short (%zd bytes), ignoring.\n", -recv_len);
        continue;
      }
      unit_id_t legacy_unit_id;
      ext_memcpy(&legacy_unit_id, msg, sizeof(unit_id_t));
      eot_handle_legacy_message(msg + sizeof(unit_id_t), -recv_len - sizeof(unit_id_t));
    } else {
      switch (state) {
      case EOT_WAIT_ADV:
        if (msg_type == HOT_MSG_ADV) {
          // received advertisement from HOT
          ext_io_puts("EOT: received advertisement from HOT, initiating connection...\n");

          timer_now(&pairing_start);
          // reset connection info from previous session
          ext_memset(&conn, 0, sizeof(conn));

          ext_io_printf("EOT: establishing connection with ID %u\n", recved_session_id);
          conn.session_id = recved_session_id;

          // generate our keypair
          if (!generate_keypair(&conn.eot_keys)) {
            ext_io_eprintf("Failed to generate EOT keypair\n");
            ext_exit(1);
          }
          // send our public key to HOT
          uint8_t compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          compress_pubkey(conn.eot_keys.public, compressed_pubkey);
          comm_send(comm, conn.session_id, (msg_type_t)EOT_MSG_PUBKEY, compressed_pubkey, sizeof(compressed_pubkey), NULL);
          ext_io_puts("EOT: sent public key to HOT, waiting for their pubkey and commitment...\n");
          state = EOT_KEY_EX_1;
        }
        break;
      case EOT_KEY_EX_1:
        if (msg_type == HOT_MSG_PUBKEY_AND_COMMIT && recved_session_id == conn.session_id && recv_len == COMPRESSED_PUBKEY_SIZE + COMMITMENT_SIZE) {
          // received HOT's public key and commitment
          uint8_t hot_compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          ext_memcpy(hot_compressed_pubkey, msg, COMPRESSED_PUBKEY_SIZE);
          ext_memcpy(conn.hot_commitment.data, msg + COMPRESSED_PUBKEY_SIZE, COMMITMENT_SIZE);
          if (!decompress_pubkey(hot_compressed_pubkey, conn.hot_keys.public)) {
            ext_io_puts("EOT: Invalid HOT public key received, aborting connection.\n");
            state = EOT_IDLE;
            break;
          }
          ext_io_puts("EOT: received HOT pubkey and commitment, generating nonce...\n");
          // generate our nonce
          generate_nonce(&conn.eot_nonce);
          // send our nonce to HOT
          comm_send(comm, conn.session_id, (msg_type_t)EOT_MSG_NONCE, conn.eot_nonce.data, sizeof(conn.eot_nonce.data), NULL);
          ext_io_puts("EOT: sent nonce to HOT, waiting for their nonce...\n");
          state = EOT_KEY_EX_2;
        }
        break;
      case EOT_KEY_EX_2:
        if (msg_type == HOT_MSG_NONCE && recved_session_id == conn.session_id && recv_len == NONCE_SIZE) {
          // received HOT's nonce
          ext_memcpy(conn.hot_nonce.data, msg, NONCE_SIZE);
          // verify HOT's commitment
          if (!verify_commitment(&conn.hot_nonce, &conn.hot_commitment)) {
            ext_io_eprintf("EOT: Commitment verification failed! Aborting.\n");
            state = EOT_IDLE;
            break;
          }
          timer_now(&now);
          ext_io_printf("EOT: pairing took %d ms\n", timer_diff_ms(&now, &pairing_start));
          // compute PIN
          conn.pin = compute_pin(conn.eot_keys.public, conn.hot_keys.public, &conn.eot_nonce, &conn.hot_nonce);
          compute_shared_secret(conn.eot_keys.private, conn.hot_keys.public, conn.shared_secret);
          ext_io_printf("EOT: received HOT nonce and verified commitment. PIN is %05u\n", conn.pin);
          ext_io_puts("Please enter the PIN in the HOT and press the TEST button once confirmed.\n");
          wait_for_arm_button_press();
          ext_io_puts("Pairing successful! Press enter at any time to disconnect and return to waiting for advertisement.\n");
          state = EOT_PAIRED;
        }
        break;
      case EOT_PAIRED:
        if (recved_session_id == conn.session_id) {
          // check message counter for replay protection
          if (recv_len < (ssize_t)sizeof(conn.ctr)) {
            ext_io_puts("EOT: received message too short for counter, ignoring.\n");
            break;
          }
          msg_ctr_t recv_ctr = *(msg_ctr_t*)msg;
          if (recv_ctr <= conn.ctr) {
            ext_io_printf("EOT: received message with old counter (%u <= %u), ignoring.\n", recv_ctr, conn.ctr);
            break;
          }
          conn.ctr = recv_ctr;
          if (msg_type == HOT_MSG_STATUS) {
            get_eot_status(&status);
            comm_send(comm, conn.session_id, (msg_type_t)EOT_MSG_STATUS, (uint8_t*)&status, sizeof(status), conn.shared_secret);
            ext_io_puts("EOT: sent status update to HOT.\n");
          } else if (msg_type == HOT_MSG_EMERGENCY) {
            eot_emergency_brake();
            comm_send(comm, conn.session_id, (msg_type_t)EOT_MSG_EMERGENCY, NULL, 0, conn.shared_secret);
            ext_io_puts("EOT: sent emergency brake confirmation to HOT.\n");
          } else if (msg_type == HOT_MSG_DISCONNECT) {
            ext_io_puts("EOT: received disconnect\n");
            state = EOT_IDLE;
            ext_memset(&conn, 0, sizeof(conn));
          }
        }
      default:
        break;
      }
    }
  }
}

void hot_run(communicator_t *comm) {
  // HOT main loop

  conn_info_t conn; // current connection info
  ext_memset(&conn, 0, sizeof(conn));
  session_id_t recved_session_id = 0;
  enum hot_state state = HOT_IDLE;
  uint8_t msg[MAX_MSG_LEN];
  msg_type_t msg_type;
  protocol_timer_t last_adv_time;
  protocol_timer_t last_transmit_time;
  protocol_timer_t pairing_start;
  protocol_timer_t now;
  timer_now(&last_adv_time);
  timer_now(&last_transmit_time);
  timer_now(&pairing_start);
  timer_now(&now);
  uint32_t input_pin = 0;
  int choice;
  uint8_t* shared_secret;
  unit_id_t recent_upgradable_legacy_unit_id = 0;
  
  while (1) {
    // use shared_secret once paired
    if (state == HOT_PAIRED || state == HOT_WAIT_FOR_STATUS || state == HOT_WAIT_FOR_EMERGENCY) {
      shared_secret = conn.shared_secret;
    } else {
      shared_secret = NULL;
    }
    ssize_t recv_len = comm_recv(comm, &recved_session_id, &msg_type, msg, sizeof(msg), shared_secret);
    timer_now(&now);
    if (recv_len == -1) {
      // timeout from recv, no message received
      switch (state) {
      case HOT_IDLE:
        ext_io_puts("HOT_IDLE: waiting for user to push ARM button\n");
        ext_io_puts("Options:\n");
        ext_io_puts("  -1: Push ARM button to start pairing\n");
        ext_io_puts("  <5-digit unit ID>: Enter legacy mode with given unit ID\n");
        ext_io_puts("Enter choice: ");
        ext_io_flush();
        ext_io_scan_int(&choice);
        if (choice == -1) {
          ext_io_puts("Button pressed, sending advertisement...\n");
          // reset connection info from previous session
          ext_memset(&conn, 0, sizeof(conn));
          // generate random session_id for this session
          conn.session_id = generate_session_id();
          ext_io_printf("session id: %u\n", conn.session_id);
          state = HOT_ADV;
        } else if (choice >= 0 && choice <= 99999) {
          if (recent_upgradable_legacy_unit_id == (unit_id_t)choice) {
            ext_io_printf("Requested to pair in legacy mode with unit ID %05u, but it supports the new protocol. Please use the new protocol to pair.\n", choice);
          } else {
            ext_io_printf("Entering legacy mode with unit ID %05u\n", choice);
            state = HOT_LEGACY;
          }
        } else {
          ext_io_puts("Invalid choice, staying in idle state.\n");
        }
        break;
      case HOT_ADV:
        if (timer_diff_ms(&now, &last_adv_time) >= HOT_ADV_INTERVAL_MS) {
          // send advertisement
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_ADV, NULL, 0, NULL);
          last_adv_time = now;
          ext_io_puts("HOT: sent advertisement\n");
        }
        break;
      case HOT_KEY_EX_1:
        if (timer_diff_ms(&now, &pairing_start) >= PAIRING_TIMEOUT) {
          ext_io_puts("HOT: Pairing timed out. Returning to idle state.\n");
          state = HOT_IDLE;
          ext_memset(&conn, 0, sizeof(conn));
        }
        break;
      case HOT_WAIT_FOR_PIN:
        // In a real implementation, we would wait for user input here
        ext_io_printf("HOT: waiting for user to input PIN (expected PIN is %05u)\n", conn.pin);
        for (int i = 0; i < 3; i++) {
          ext_io_printf("Enter PIN (attempt %d of 3): ", i + 1);
          ext_io_flush();
          ext_io_scan_uint(&input_pin);
          ext_io_printf("You entered: %05u\n", input_pin);
          if (input_pin == conn.pin) {
            ext_io_puts("PIN correct! Pairing successful. Press the ARM button on the EOT to confirm.\n");
            state = HOT_PAIRED;
            break;
          } else {
            ext_io_puts("Incorrect PIN. Try again.\n");
          }
        }
        if (state != HOT_PAIRED) {
          ext_io_puts("Failed to enter correct PIN. Returning to idle state.\n");
          state = HOT_IDLE;
          ext_memset(&conn, 0, sizeof(conn));
        }
        break;
      case HOT_PAIRED:
        ext_io_puts("Select an option:\n");
        ext_io_puts("1. Request status update\n");
        ext_io_puts("2. Emergency brake\n");
        ext_io_puts("3. Disconnect (ARM button)\n");
        ext_io_puts("Enter choice: ");
        ext_io_flush();
        choice = 0;
        ext_io_scan_int(&choice);

        conn.ctr++; // increment message counter
        if (choice == 1) {
          // also send ctr to avoid replay attacks
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_STATUS, (uint8_t*)&conn.ctr, sizeof(conn.ctr), conn.shared_secret);
          ext_io_puts("HOT: sent status update request\n");
          state = HOT_WAIT_FOR_STATUS;
        } else if (choice == 2) {
          // also send ctr to avoid replay attacks
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_EMERGENCY, (uint8_t*)&conn.ctr, sizeof(conn.ctr), conn.shared_secret);
          ext_io_puts("HOT: sent emergency brake request\n");
          state = HOT_WAIT_FOR_EMERGENCY;
        } else if (choice == 3) {
          // also send ctr to avoid replay attacks
          ext_io_puts("HOT: Disconnecting...\n");
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_DISCONNECT, (uint8_t*)&conn.ctr, sizeof(conn.ctr), conn.shared_secret);
          state = HOT_IDLE;
          ext_memset(&conn, 0, sizeof(conn));
        } else {
          ext_io_puts("Invalid choice.\n");
        }
        timer_now(&last_transmit_time);
        break;
      case HOT_WAIT_FOR_STATUS:
      case HOT_WAIT_FOR_EMERGENCY:
        // retransmit request if no response
        conn.ctr++; // increment message counter
        if (timer_diff_ms(&now, &last_transmit_time) >= HOT_RETRANSMIT_INTERVAL_MS) {
          if (state == HOT_WAIT_FOR_STATUS) {
            comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_STATUS, (uint8_t*)&conn.ctr, sizeof(conn.ctr), conn.shared_secret);
            ext_io_puts("HOT: retransmitted status update request\n");
          } else if (state == HOT_WAIT_FOR_EMERGENCY) {
            comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_EMERGENCY, (uint8_t*)&conn.ctr, sizeof(conn.ctr), conn.shared_secret);
            ext_io_puts("HOT: retransmitted emergency brake request\n");
          }
          last_transmit_time = now;
        }
        break;
      default:
        break;
      }
    } else if (recv_len == -2) {
      // signature verification failed
      ext_io_puts("HOT: received message with invalid signature, ignoring.\n");
    } else if (recv_len < 0) {
      // legacy message received
      if (state != HOT_LEGACY) {
        ext_io_puts("HOT: received legacy message while not in legacy state, ignoring.\n");
        continue;
      }
      if (-recv_len < (ssize_t)sizeof(unit_id_t)) {
        ext_io_eprintf("HOT: received legacy message too short (%zd bytes), ignoring.\n", -recv_len);
        continue;
      }
      unit_id_t legacy_unit_id;
      ext_memcpy(&legacy_unit_id, msg, sizeof(unit_id_t));
      if (recent_upgradable_legacy_unit_id != 0 && legacy_unit_id == recent_upgradable_legacy_unit_id) {
        ext_io_printf("HOT: received legacy message from recently upgradable unit ID %u, ignoring to prevent downgrade attack.\n", legacy_unit_id);
      } else if ((size_t)-recv_len > sizeof(unit_id_t) + 3 && ext_memcmp(msg + sizeof(unit_id_t), "ARM", 3) == 0) {
        ext_io_printf("HOT: received legacy ARM command from unit ID %u, entering legacy ARMED mode.\n", legacy_unit_id);
        state = HOT_LEGACY_ARMED;
      } else if (state == HOT_LEGACY_ARMED) {
        hot_handle_legacy_message(msg + sizeof(unit_id_t), -recv_len - sizeof(unit_id_t));
      } else {
        ext_io_printf("HOT: received legacy message from unit ID %u while not in legacy state, ignoring.\n", legacy_unit_id);
      }
    } else {
      if (msg_type == EOT_MSG_UPGRADE && recv_len == (ssize_t)sizeof(unit_id_t)) {
        ext_memcpy(&recent_upgradable_legacy_unit_id, msg, sizeof(unit_id_t));
        ext_io_printf("HOT: received protocol upgrade request from legacy unit ID %u\n", recent_upgradable_legacy_unit_id);
      }
      switch (state) {
      case HOT_ADV:
        if (msg_type == EOT_MSG_PUBKEY && recved_session_id == conn.session_id) {
          ext_io_puts("HOT: received EOT pubkey, initiating connection...\n");

          timer_now(&pairing_start);
          
          uint8_t eot_compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          ext_memcpy(eot_compressed_pubkey, msg, COMPRESSED_PUBKEY_SIZE);
          decompress_pubkey(eot_compressed_pubkey, conn.eot_keys.public);

          // generate our keypair, nonce, and commitment
          if (!generate_keypair(&conn.hot_keys)) {
            ext_io_eprintf("Failed to generate HOT keypair\n");
            ext_exit(1);
          }
          generate_nonce(&conn.hot_nonce);
          create_commitment(&conn.hot_nonce, &conn.hot_commitment);
          // send our public key and commitment to EOT
          uint8_t compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          compress_pubkey(conn.hot_keys.public, compressed_pubkey);
          uint8_t payload[COMPRESSED_PUBKEY_SIZE + COMMITMENT_SIZE];
          ext_memcpy(payload, compressed_pubkey, COMPRESSED_PUBKEY_SIZE);
          ext_memcpy(payload + COMPRESSED_PUBKEY_SIZE, conn.hot_commitment.data, COMMITMENT_SIZE);
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_PUBKEY_AND_COMMIT, payload, sizeof(payload), NULL);

          ext_io_puts("HOT: sent pubkey and commitment to EOT, waiting for their nonce...\n");
          state = HOT_KEY_EX_1;
        }
        break;
      case HOT_KEY_EX_1:
        if (msg_type == EOT_MSG_NONCE && recved_session_id == conn.session_id && recv_len == NONCE_SIZE) {
          ext_io_puts("HOT: received EOT nonce, sending our nonce...\n");
          // can also generate the expected PIN here
          ext_memcpy(conn.eot_nonce.data, msg, NONCE_SIZE);
          conn.pin = compute_pin(conn.eot_keys.public, conn.hot_keys.public, &conn.eot_nonce, &conn.hot_nonce);
          compute_shared_secret(conn.hot_keys.private, conn.eot_keys.public, conn.shared_secret);
          // send our nonce to EOT
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_NONCE, conn.hot_nonce.data, sizeof(conn.hot_nonce.data), NULL);
          ext_io_puts("HOT: sent nonce to EOT, waiting for user to input PIN...\n");
          state = HOT_WAIT_FOR_PIN;
        }
      case HOT_WAIT_FOR_STATUS:
        if (msg_type == EOT_MSG_STATUS && recved_session_id == conn.session_id && recv_len == (ssize_t)sizeof(eot_status_t)) {
          eot_status_t status;
          ext_memcpy(&status, msg, sizeof(eot_status_t));
          display_eot_status(&status);
          state = HOT_PAIRED;
        }
        break;
      case HOT_WAIT_FOR_EMERGENCY:
        if (msg_type == EOT_MSG_EMERGENCY && recved_session_id == conn.session_id) {
          ext_io_printf("HOT: received emergency brake confirmation from EOT. %d ms elapsed since last request.\n", timer_diff_ms(&now, &last_transmit_time));
          state = HOT_PAIRED;
        }
        break;
      default:
        break;
      }
    }
  }
}

void init_communicator(communicator_t *comm,
                         comm_device_type_t device_type,
                         const uint32_t timeout_ms) {
  comm->timeout_ms = timeout_ms;
  comm->comm_h = comm_init(device_type, timeout_ms);
  if (!comm->comm_h) {
    ext_io_eprintf("Failed to initialize communication\n");
    ext_exit(1);
  }
}

int eot_main() {
  communicator_t comm;
  unit_id_t sample_unit_id = 12345;
  
  // Initialize abstraction layers
  ext_timer_init();
  ext_io_init();
  ext_random_init();
  
  ext_io_puts("EOT starting...\n");
  init_communicator(&comm, COMM_DEVICE_EOT, DEFAULT_TIMEOUT_MS);
  eot_run(&comm, sample_unit_id);
  return 0;
}

int hot_main() {
  communicator_t comm;
  
  // Initialize abstraction layers
  ext_timer_init();
  ext_io_init();
  ext_random_init();
  
  ext_io_puts("HOT starting...\n");
  init_communicator(&comm, COMM_DEVICE_HOT, DEFAULT_TIMEOUT_MS);
  hot_run(&comm);
  return 0;
}

// for testing packet dropping
void add_drop_packet(int pkt_num) {
  // pkt_num is 1-indexed
  for (size_t i = 0; i < sizeof(pkt_dropped)/sizeof(pkt_dropped[0]); i++) {
    if (pkt_dropped[i] == 0) {
      pkt_dropped[i] = pkt_num;
      return;
    }
  }

  ext_io_eprintf("Too many packets to drop, increase pkt_dropped array size\n");
  ext_exit(1);
}
