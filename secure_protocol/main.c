#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <time.h>

#include "micro-ecc/uECC.h"
#include "sha256/sha256.h"


#define EOT_TO_HOT_SOCKET_PATH "/tmp/eot_to_hot.sock"
#define HOT_TO_EOT_SOCKET_PATH "/tmp/hot_to_eot.sock"

#define DEFAULT_TIMEOUT_MS 300
#define HOT_ADV_INTERVAL_MS 1000
#define PAIRING_TIMEOUT 6000
#define PACKET_SEND_DELAY_MS_PER_BYTE 20
#define HOT_RETRANSMIT_INTERVAL_MS 2000
#define EOT_WAIT_ADV_TIMEOUT_MS 30000

#define ECC_CURVE uECC_secp256r1()
// all sizes are in bytes
#define CURVE_SIZE 32
#define PUBKEY_SIZE (CURVE_SIZE * 2)
#define COMPRESSED_PUBKEY_SIZE (CURVE_SIZE + 1)
#define PRIVKEY_SIZE CURVE_SIZE

#define NONCE_SIZE 16
#define SHA256_SIZE 32
#define COMMITMENT_SIZE 32 // SHA-256 output size

#define MAX_MSG_LEN 256
#define MAX_PKT_LEN 256

typedef uint32_t session_id_t;

enum eot_msg {
  // these are messages sent from EOT to HOT
  EOT_MSG_PUBKEY = 0,
  EOT_MSG_NONCE,
  EOT_MSG_STATUS,    // status update response
  EOT_MSG_EMERGENCY, // emergency brake confirmation
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
  EOT_EMERGENCY_BRAKE
};

enum hot_state {
  HOT_IDLE = 0,     // initial state
  HOT_ADV,          // sending periodic advertisements
  HOT_KEY_EX_1,     // received their pubkey, sending our pubkey and commitment, waiting for their nonce
  HOT_WAIT_FOR_PIN, // received their nonce, waiting for user to input PIN
  HOT_PAIRED,
  HOT_WAIT_FOR_STATUS,
  HOT_WAIT_FOR_EMERGENCY, // wait for confirmation of emergency brake
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

typedef struct {
  uint32_t timeout_ms;
  int send_fd;
  int recv_fd;
  struct sockaddr_un send_addr;
} communicator_t;

typedef struct {
  session_id_t session_id;
  keypair_t eot_keys;
  keypair_t hot_keys;
  // eot_commitment not needed, as HOT is the first sender
  commitment_t hot_commitment;
  nonce_t eot_nonce;
  nonce_t hot_nonce;
  pin_t pin;
  uint32_t ctr; // current message counter for replay protection (only used once paired)
} conn_info_t;

typedef struct timespec timer_t;

void timer_now(timer_t *t) {
  clock_gettime(CLOCK_MONOTONIC, t);
}

int timer_diff_ms(const timer_t *end, const timer_t *start) {
  return (end->tv_sec - start->tv_sec) * 1000 + (end->tv_nsec - start->tv_nsec) / 1000000;
}


void sha256(const uint8_t *data, size_t len, uint8_t *hash_out) {
  /* printf("SHA256 input (len=%zu): ", len); for (size_t i = 0; i < len; i++) printf("%02x", data[i]); printf("\n"); */

  struct sha256_buff buff;
  sha256_init(&buff);
  sha256_update(&buff, data, len);
  sha256_finalize(&buff);
  sha256_read(&buff, hash_out);

  /* printf("SHA256 output: "); for (size_t i = 0; i < SHA256_SIZE; i++) printf("%02x", hash_out[i]); printf("\n"); */
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
  return memcmp(computed_commitment.data, commitment->data, COMMITMENT_SIZE) == 0;
}

void compress_pubkey(const uint8_t *pubkey, uint8_t *compressed) {
  uECC_compress(pubkey, compressed, ECC_CURVE);
}

void decompress_pubkey(const uint8_t *compressed, uint8_t *pubkey) {
  uECC_decompress(compressed, pubkey, ECC_CURVE);
}

/**
 * Generate a random nonce.
 * 
 * @param nonce Pointer to the nonce_t struct to hold the generated nonce.
 */
void generate_nonce(nonce_t *nonce) {
  int fd = open("/dev/urandom", O_RDONLY);
  if (fd < 0) {
    perror("open /dev/urandom");
    exit(EXIT_FAILURE);
  }
  if (read(fd, nonce->data, NONCE_SIZE) != NONCE_SIZE) {
    perror("read /dev/urandom");
    close(fd);
    exit(EXIT_FAILURE);
  }
  close(fd);
}

session_id_t generate_session_id() {
  session_id_t session_id;
  int fd = open("/dev/urandom", O_RDONLY);
  if (fd < 0) {
    perror("open /dev/urandom");
    exit(EXIT_FAILURE);
  }
  if (read(fd, &session_id, sizeof(session_id)) != sizeof(session_id)) {
    perror("read /dev/urandom");
    close(fd);
    exit(EXIT_FAILURE);
  }
  close(fd);
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
  memcpy(data, eot_pubkey, PUBKEY_SIZE);
  memcpy(data + PUBKEY_SIZE, hot_pubkey, PUBKEY_SIZE);
  memcpy(data + PUBKEY_SIZE * 2, eot_nonce->data, NONCE_SIZE);
  memcpy(data + PUBKEY_SIZE * 2 + NONCE_SIZE, hot_nonce->data, NONCE_SIZE);
  uint8_t hash[SHA256_SIZE];
  sha256(data, sizeof(data), hash);
  // take first 4 bytes of hash, convert to integer, mod 100000 to get 5-digit PIN
  uint32_t pin = (*(uint32_t*)hash) % 100000;
  return pin;
}


void comm_send(communicator_t* comm,
          const session_id_t session_id,
          const msg_type_t msg_type,
          const uint8_t *msg,
          const size_t msg_len) {
  size_t total_len = sizeof(session_id_t) + sizeof(msg_type_t) + msg_len;
  if (total_len > MAX_PKT_LEN) {
    fprintf(stderr, "Message too long to send (%zu bytes)\n", total_len);
    exit(EXIT_FAILURE);
  }
  uint8_t buffer[MAX_PKT_LEN];

  memcpy(buffer, &session_id, sizeof(session_id_t));
  memcpy(buffer + sizeof(session_id_t), &msg_type, sizeof(msg_type_t));
  memcpy(buffer + sizeof(session_id_t) + sizeof(msg_type_t), msg, msg_len);

  printf("[INFO] sending message of length %zu (session_id=%u, msg_type=%d)\n", total_len, session_id, msg_type);
  // simulate a packet delay in sending
  usleep(total_len * PACKET_SEND_DELAY_MS_PER_BYTE * 1000);

  if (sendto(comm->send_fd, buffer, total_len, 0,
             (struct sockaddr*)&comm->send_addr, sizeof(comm->send_addr)) != total_len) {
    perror("sendto");
    exit(EXIT_FAILURE);
  }
}

ssize_t comm_recv(communicator_t* comm,
          session_id_t *session_id,
          msg_type_t *msg_type,
          uint8_t *msg,
          const size_t max_msg_len) {
  uint8_t buffer[MAX_PKT_LEN];
  size_t header_len = sizeof(session_id_t) + sizeof(msg_type_t);

  ssize_t recv_len = recvfrom(comm->recv_fd, buffer, sizeof(buffer), 0, NULL, NULL);
  if (recv_len < 0) {
    if (recv_len == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      // timeout
      return -1;
    }
    perror("recvfrom");
    exit(EXIT_FAILURE);
  }

  size_t payload_len = recv_len - header_len;

  if (payload_len > max_msg_len) {
    fprintf(stderr, "Received message too long (data: %zu bytes, total: %zd bytes)\n", payload_len, recv_len);
    exit(EXIT_FAILURE);
  }

  memcpy(session_id, buffer, sizeof(session_id_t));
  memcpy(msg_type, buffer + sizeof(session_id_t), sizeof(msg_type_t));
  memcpy(msg, buffer + header_len, payload_len);

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
  time_t now;
  char buf[26];
  time(&now);
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));

  printf("[%s] EOT Status:\n", buf);
  printf("  Battery Condition: %u\n", status->batt_cond);
  printf("  Pressure: %u\n", status->pressure);
  printf("  Battery Charge Used: %u\n", status->batt_charge_used);
  printf("  Valve Circuit Operational: %u\n", status->valve_circuit_operational);
  printf("  Confirmation Indicator: %u\n", status->confirmation_indicator);
  printf("  Turbine Status: %u\n", status->turbine_status);
  printf("  Motion Detection: %u\n", status->motion_detection);
  printf("  Marker Light Battery Weak: %u\n", status->marker_light_battery_weak);
  printf("  Marker Light Status: %u\n", status->marker_light_status);
  printf("\n");
}

void eot_emergency_brake() {
  // In a real implementation, trigger the emergency brake mechanism
  printf("EOT: Emergency brake activated!\n");
}

/**
 * Simulate waiting for the user to press the ARM button.
 * Uses STDIN for simulation.
 */
void wait_for_arm_button_press() {
  // clear any existing input
  int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
  char buf[16];
  while (read(STDIN_FILENO, buf, sizeof(buf)) > 0);
  fcntl(STDIN_FILENO, F_SETFL, flags);
  
  printf("Press enter to simulate ARM button press...\n");
  getchar();
}

void eot_run(communicator_t *comm) {
  // EOT main loop

  conn_info_t conn; // current connection info
  memset(&conn, 0, sizeof(conn));
  session_id_t recved_session_id = 0;
  enum eot_state state = EOT_IDLE;
  uint8_t msg[MAX_MSG_LEN];
  msg_type_t msg_type;
  eot_status_t status;
  timer_t adv_start;
  timer_t pairing_start;
  timer_t now;
  timer_now(&adv_start);
  timer_now(&pairing_start);
  timer_now(&now);

  while (1) {
    size_t recv_len = comm_recv(comm, &recved_session_id, &msg_type, msg, sizeof(msg));
    if (recv_len == -1) {
      // timeout from recv, no message received
      switch (state) {
      case EOT_IDLE:
        printf("EOT_IDLE: waiting for user to push ARM button\n");
        wait_for_arm_button_press();
        printf("Button pressed, waiting for HOT advertisement...\n");
        timer_now(&adv_start);
        state = EOT_WAIT_ADV;
        break;
      case EOT_WAIT_ADV:
        timer_now(&now);
        if (timer_diff_ms(&now, &adv_start) >= EOT_WAIT_ADV_TIMEOUT_MS) {
          printf("EOT: Waiting for HOT timed out. Returning to idle state.\n");
          state = EOT_IDLE;
          memset(&conn, 0, sizeof(conn));
        }
        break;
      case EOT_KEY_EX_1:
      case EOT_KEY_EX_2:
        timer_now(&now);
        if (timer_diff_ms(&now, &pairing_start) >= PAIRING_TIMEOUT) {
          printf("EOT: Pairing timed out. Returning to idle state.\n");
          state = EOT_IDLE;
          memset(&conn, 0, sizeof(conn));
        }
        break;
      case EOT_PAIRED:
        // TODO: periodically send status updates
        // check for user input (non-blocking) to disconnect
        {
          int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
          fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
          char buf[16];
          if (read(STDIN_FILENO, buf, sizeof(buf)) > 0) {
            printf("Pressed enter (ARM button), disconnecting and searching for new HOT...\n");
            timer_now(&adv_start);
            state = EOT_WAIT_ADV;
            memset(&conn, 0, sizeof(conn));
          }
          fcntl(STDIN_FILENO, F_SETFL, flags);
        }
        break;
      default:
        break;
      }
    } else {
      /* printf("received msg type=%d session_id=%u len=%zu state=%d\n", msg_type, recved_session_id, recv_len, state); */
      switch (state) {
      case EOT_WAIT_ADV:
        if (msg_type == HOT_MSG_ADV) {
          // received advertisement from HOT
          printf("EOT: received advertisement from HOT, initiating connection...\n");

          timer_now(&pairing_start);
          // reset connection info from previous session
          memset(&conn, 0, sizeof(conn));

          printf("EOT: establishing connection with ID %u\n", recved_session_id);
          conn.session_id = recved_session_id;

          // generate our keypair
          if (!generate_keypair(&conn.eot_keys)) {
            fprintf(stderr, "Failed to generate EOT keypair\n");
            exit(EXIT_FAILURE);
          }
          // send our public key to HOT
          uint8_t compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          compress_pubkey(conn.eot_keys.public, compressed_pubkey);
          comm_send(comm, conn.session_id, (msg_type_t)EOT_MSG_PUBKEY, compressed_pubkey, sizeof(compressed_pubkey));
          printf("EOT: sent public key to HOT, waiting for their pubkey and commitment...\n");
          state = EOT_KEY_EX_1;
        }
        break;
      case EOT_KEY_EX_1:
        if (msg_type == HOT_MSG_PUBKEY_AND_COMMIT && recved_session_id == conn.session_id && recv_len == COMPRESSED_PUBKEY_SIZE + COMMITMENT_SIZE) {
          // received HOT's public key and commitment
          uint8_t hot_compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          memcpy(hot_compressed_pubkey, msg, COMPRESSED_PUBKEY_SIZE);
          memcpy(conn.hot_commitment.data, msg + COMPRESSED_PUBKEY_SIZE, COMMITMENT_SIZE);
          decompress_pubkey(hot_compressed_pubkey, conn.hot_keys.public);
          printf("EOT: received HOT pubkey and commitment, generating nonce...\n");
          // generate our nonce
          generate_nonce(&conn.eot_nonce);
          // send our nonce to HOT
          comm_send(comm, conn.session_id, (msg_type_t)EOT_MSG_NONCE, conn.eot_nonce.data, sizeof(conn.eot_nonce.data));
          printf("EOT: sent nonce to HOT, waiting for their nonce...\n");
          state = EOT_KEY_EX_2;
        }
        break;
      case EOT_KEY_EX_2:
        if (msg_type == HOT_MSG_NONCE && recved_session_id == conn.session_id && recv_len == NONCE_SIZE) {
          // received HOT's nonce
          memcpy(conn.hot_nonce.data, msg, NONCE_SIZE);
          // verify HOT's commitment
          if (!verify_commitment(&conn.hot_nonce, &conn.hot_commitment)) {
            fprintf(stderr, "EOT: Commitment verification failed! Aborting.\n");
            state = EOT_IDLE;
            break;
          }
          timer_now(&now);
          printf("EOT: pairing took %d ms\n", timer_diff_ms(&now, &pairing_start));
          // compute PIN
          conn.pin = compute_pin(conn.eot_keys.public, conn.hot_keys.public, &conn.eot_nonce, &conn.hot_nonce);
          printf("EOT: received HOT nonce and verified commitment. PIN is %05u\n", conn.pin);
          printf("Please enter the PIN in the HOT and press the ARM button once confirmed.\n");
          wait_for_arm_button_press();
          printf("Pairing successful! Press enter at any time to disconnect and return to waiting for advertisement.\n");
          state = EOT_PAIRED;
        }
        break;
      case EOT_PAIRED:
        if (recved_session_id == conn.session_id) {
          // check message counter for replay protection
          if (recv_len < sizeof(conn.ctr)) {
            printf("EOT: received message too short for counter, ignoring.\n");
            break;
          }
          uint32_t recv_ctr;
          memcpy(&recv_ctr, msg, sizeof(recv_ctr));
          if (recv_ctr <= conn.ctr) {
            printf("EOT: received message with old counter (%u <= %u), ignoring.\n", recv_ctr, conn.ctr);
            break;
          }
          conn.ctr = recv_ctr;
          if (msg_type == HOT_MSG_STATUS) {
            get_eot_status(&status);
            comm_send(comm, conn.session_id, (msg_type_t)EOT_MSG_STATUS, (uint8_t*)&status, sizeof(status));
            printf("EOT: sent status update to HOT.\n");
          } else if (msg_type == HOT_MSG_EMERGENCY) {
            eot_emergency_brake();
            comm_send(comm, conn.session_id, (msg_type_t)EOT_MSG_EMERGENCY, NULL, 0);
            printf("EOT: sent emergency brake confirmation to HOT.\n");
          } else if (msg_type == HOT_MSG_DISCONNECT) {
            printf("EOT: received disconnect\n");
            state = EOT_IDLE;
            memset(&conn, 0, sizeof(conn));
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
  memset(&conn, 0, sizeof(conn));
  session_id_t recved_session_id = 0;
  enum hot_state state = HOT_IDLE;
  uint8_t msg[MAX_MSG_LEN];
  msg_type_t msg_type;
  timer_t last_adv_time;
  timer_t last_transmit_time;
  timer_t pairing_start;
  timer_t now;
  timer_now(&last_adv_time);
  timer_now(&last_transmit_time);
  timer_now(&pairing_start);
  timer_now(&now);
  uint32_t input_pin = 0;
  int choice;
  
  while (1) {
    size_t recv_len = comm_recv(comm, &recved_session_id, &msg_type, msg, sizeof(msg));
    if (recv_len == -1) {
      // timeout from recv, no message received
      switch (state) {
      case HOT_IDLE:
        printf("HOT_IDLE: waiting for user to push ARM button\n");
        wait_for_arm_button_press();
        printf("Button pressed, sending advertisement...\n");
        // reset connection info from previous session
        memset(&conn, 0, sizeof(conn));
        // generate random session_id for this session
        conn.session_id = generate_session_id();
        printf("session id: %u\n", conn.session_id);
        state = HOT_ADV;
        break;
      case HOT_ADV:
        timer_now(&now);
        if (timer_diff_ms(&now, &last_adv_time) >= HOT_ADV_INTERVAL_MS) {
          // send advertisement
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_ADV, NULL, 0);
          last_adv_time = now;
          printf("HOT: sent advertisement\n");
        }
        break;
      case HOT_KEY_EX_1:
        timer_now(&now);
        if (timer_diff_ms(&now, &pairing_start) >= PAIRING_TIMEOUT) {
          printf("HOT: Pairing timed out. Returning to idle state.\n");
          state = HOT_IDLE;
          memset(&conn, 0, sizeof(conn));
        }
        break;
      case HOT_WAIT_FOR_PIN:
        // In a real implementation, we would wait for user input here
        printf("HOT: waiting for user to input PIN...\n");
        for (int i = 0; i < 3; i++) {
          printf("Enter PIN (attempt %d of 3): ", i + 1);
          scanf("%u", &input_pin);
          getchar(); // consume newline
          printf("You entered: %05u\n", input_pin);
          if (input_pin == conn.pin) {
            printf("PIN correct! Pairing successful. Press the ARM button on the EOT to confirm.\n");
            state = HOT_PAIRED;
            break;
          } else {
            printf("Incorrect PIN. Try again.\n");
          }
        }
        if (state != HOT_PAIRED) {
          printf("Failed to enter correct PIN. Returning to idle state.\n");
          state = HOT_IDLE;
          memset(&conn, 0, sizeof(conn));
        }
        break;
      case HOT_PAIRED:
        printf("Select an option:\n");
        printf("1. Request status update\n");
        printf("2. Emergency brake\n");
        printf("3. Disconnect (ARM button)\n");
        printf("Enter choice: ");
        choice = 0;
        scanf("%d", &choice);
        getchar(); // consume newline

        conn.ctr++; // increment message counter
        if (choice == 1) {
          // also send ctr to avoid replay attacks
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_STATUS, (uint8_t*)&conn.ctr, sizeof(conn.ctr));
          printf("HOT: sent status update request\n");
          state = HOT_WAIT_FOR_STATUS;
        } else if (choice == 2) {
          // also send ctr to avoid replay attacks
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_EMERGENCY, (uint8_t*)&conn.ctr, sizeof(conn.ctr));
          printf("HOT: sent emergency brake request\n");
          state = HOT_WAIT_FOR_EMERGENCY;
        } else if (choice == 3) {
          // also send ctr to avoid replay attacks
          printf("HOT: Disconnecting...\n");
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_DISCONNECT, (uint8_t*)&conn.ctr, sizeof(conn.ctr));
          state = HOT_IDLE;
          memset(&conn, 0, sizeof(conn));
        } else {
          printf("Invalid choice.\n");
        }
        timer_now(&last_transmit_time);
        break;
      case HOT_WAIT_FOR_STATUS:
      case HOT_WAIT_FOR_EMERGENCY:
        // retransmit request if no response
        timer_now(&now);
        if (timer_diff_ms(&now, &last_transmit_time) >= HOT_RETRANSMIT_INTERVAL_MS) {
          if (state == HOT_WAIT_FOR_STATUS) {
            comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_STATUS, (uint8_t*)&conn.ctr, sizeof(conn.ctr));
            printf("HOT: retransmitted status update request\n");
          } else if (state == HOT_WAIT_FOR_EMERGENCY) {
            comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_EMERGENCY, (uint8_t*)&conn.ctr, sizeof(conn.ctr));
            printf("HOT: retransmitted emergency brake request\n");
          }
          last_transmit_time = now;
        }
        break;
      default:
        break;
      }
    } else {
      switch (state) {
      case HOT_ADV:
        if (msg_type == EOT_MSG_PUBKEY && recved_session_id == conn.session_id) {
          printf("HOT: received EOT pubkey, initiating connection...\n");

          timer_now(&pairing_start);
          
          uint8_t eot_compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          memcpy(eot_compressed_pubkey, msg, COMPRESSED_PUBKEY_SIZE);
          decompress_pubkey(eot_compressed_pubkey, conn.eot_keys.public);

          // generate our keypair, nonce, and commitment
          if (!generate_keypair(&conn.hot_keys)) {
            fprintf(stderr, "Failed to generate HOT keypair\n");
            exit(EXIT_FAILURE);
          }
          generate_nonce(&conn.hot_nonce);
          create_commitment(&conn.hot_nonce, &conn.hot_commitment);
          // send our public key and commitment to EOT
          uint8_t compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          compress_pubkey(conn.hot_keys.public, compressed_pubkey);
          uint8_t payload[COMPRESSED_PUBKEY_SIZE + COMMITMENT_SIZE];
          memcpy(payload, compressed_pubkey, COMPRESSED_PUBKEY_SIZE);
          memcpy(payload + COMPRESSED_PUBKEY_SIZE, conn.hot_commitment.data, COMMITMENT_SIZE);
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_PUBKEY_AND_COMMIT, payload, sizeof(payload));

          printf("HOT: sent pubkey and commitment to EOT, waiting for their nonce...\n");
          state = HOT_KEY_EX_1;
        }
        break;
      case HOT_KEY_EX_1:
        if (msg_type == EOT_MSG_NONCE && recved_session_id == conn.session_id && recv_len == NONCE_SIZE) {
          printf("HOT: received EOT nonce, sending our nonce...\n");
          // can also generate the expected PIN here
          memcpy(conn.eot_nonce.data, msg, NONCE_SIZE);
          conn.pin = compute_pin(conn.eot_keys.public, conn.hot_keys.public, &conn.eot_nonce, &conn.hot_nonce);
          // send our nonce to EOT
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_NONCE, conn.hot_nonce.data, sizeof(conn.hot_nonce.data));
          printf("HOT: sent nonce to EOT, waiting for user to input PIN...\n");
          state = HOT_WAIT_FOR_PIN;
        }
      case HOT_WAIT_FOR_STATUS:
        if (msg_type == EOT_MSG_STATUS && recved_session_id == conn.session_id && recv_len == sizeof(eot_status_t)) {
          eot_status_t status;
          memcpy(&status, msg, sizeof(eot_status_t));
          display_eot_status(&status);
          state = HOT_PAIRED;
        }
        break;
      case HOT_WAIT_FOR_EMERGENCY:
        if (msg_type == EOT_MSG_EMERGENCY && recved_session_id == conn.session_id) {
          printf("HOT: received emergency brake confirmation from EOT.\n");
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
                         const char *send_socket_path,
                         const char *recv_socket_path,
                         const uint32_t timeout_ms) {
  comm->timeout_ms = timeout_ms;

  // create send socket
  comm->send_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
  if (comm->send_fd < 0) {
    perror("socket send");
    exit(EXIT_FAILURE);
  }

  memset(&comm->send_addr, 0, sizeof(comm->send_addr));
  comm->send_addr.sun_family = AF_UNIX;
  strncpy(comm->send_addr.sun_path, send_socket_path, sizeof(comm->send_addr.sun_path) - 1);

  // create recv socket
  comm->recv_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
  if (comm->recv_fd < 0) {
    perror("socket recv");
    exit(EXIT_FAILURE);
  }

  struct sockaddr_un recv_addr;
  memset(&recv_addr, 0, sizeof(recv_addr));
  recv_addr.sun_family = AF_UNIX;
  strncpy(recv_addr.sun_path, recv_socket_path, sizeof(recv_addr.sun_path) - 1);

  unlink(recv_socket_path); // remove existing socket file

  if (bind(comm->recv_fd, (struct sockaddr*)&recv_addr, sizeof(recv_addr)) < 0) {
    perror("bind recv");
    exit(EXIT_FAILURE);
  }

  // set recv timeout
  struct timeval tv;
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  if (setsockopt(comm->recv_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
    perror("setsockopt");
    exit(EXIT_FAILURE);
  }
}

int eot_main() {
  communicator_t comm;
  printf("EOT starting...\n");
  init_communicator(&comm, EOT_TO_HOT_SOCKET_PATH, HOT_TO_EOT_SOCKET_PATH, DEFAULT_TIMEOUT_MS);
  eot_run(&comm);
  return 0;
}

int hot_main() {
  communicator_t comm;
  printf("HOT starting...\n");
  init_communicator(&comm, HOT_TO_EOT_SOCKET_PATH, EOT_TO_HOT_SOCKET_PATH, DEFAULT_TIMEOUT_MS);
  hot_run(&comm);
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s [eot|hot]\n", argv[0]);
    return EXIT_FAILURE;
  }

  if (strcmp(argv[1], "eot") == 0) {
    return eot_main();
  } else if (strcmp(argv[1], "hot") == 0) {
    return hot_main();
  } else {
    fprintf(stderr, "Invalid argument: %s. Use 'eot' or 'hot'.\n", argv[1]);
    return EXIT_FAILURE;
  }

  return 0;
}
