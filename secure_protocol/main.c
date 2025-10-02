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


#define EOT_TO_HOT_SOCKET_PATH "/tmp/eot_to_hot.sock"
#define HOT_TO_EOT_SOCKET_PATH "/tmp/hot_to_eot.sock"

#define DEFAULT_TIMEOUT_MS 300
#define HOT_STATE_ADV_INTERVAL_MS 1000

#define ECC_CURVE uECC_secp256r1()
// all sizes are in bytes
#define CURVE_SIZE 32
#define PUBKEY_SIZE (CURVE_SIZE * 2)
#define COMPRESSED_PUBKEY_SIZE (CURVE_SIZE + 1)
#define PRIVKEY_SIZE CURVE_SIZE

#define NONCE_SIZE 16
#define COMMITMENT_SIZE 32 // SHA-256 output size

#define MAX_MSG_LEN 256
#define MAX_PKT_LEN 256

typedef uint32_t session_id_t;

typedef union {
  enum {
    // these are messages sent from EOT to HOT
    EOT_MSG_PUBKEY = 0,
    EOT_MSG_NONCE,
  } eot;
  enum {
    // these are messages sent from HOT to EOT
    HOT_MSG_ADV = 0,
    HOT_MSG_PUBKEY_AND_COMMIT,
    HOT_MSG_NONCE,
  } hot;
} msg_type_t;

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
} conn_info_t;

enum eot_state {
  EOT_STATE_IDLE = 0,   // initial state
  EOT_STATE_WAIT_ADV,   // waiting for advertisement from HOT
  EOT_STATE_KEY_EX_1,   // sent our pubkey, waiting for theirs
  EOT_STATE_KEY_EX_2,   // received their pubkey and commitment, sending our nonce and waiting for theirs
  EOT_STATE_PIN_DISPLAY // received their nonce, displaying PIN and waiting for user to confirm
};

enum hot_state {
  HOT_STATE_IDLE = 0,    // initial state
  HOT_STATE_ADV,         // sending periodic advertisements
  HOT_STATE_KEY_EX_1,    // received their pubkey, sending our pubkey and commitment, waiting for their nonce
  HOT_STATE_KEY_EX_2,    // received their nonce, sending our nonce
  HOT_STATE_WAIT_FOR_PIN // sent our nonce, waiting for user to input PIN
};


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
  // Simple SHA-256 hash for commitment
  // In a real implementation, use a proper cryptographic library
  // Here we just simulate it with a placeholder
  for (size_t i = 0; i < COMMITMENT_SIZE; i++) {
    commitment->data[i] = nonce->data[i % NONCE_SIZE] ^ (uint8_t)i;
  }
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

void eot_run(communicator_t *comm) {
  // EOT main loop

  conn_info_t conn; // current connection info
  memset(&conn, 0, sizeof(conn));
  enum eot_state state = EOT_STATE_IDLE;
  uint8_t msg[MAX_MSG_LEN];
  msg_type_t msg_type;

  while (1) {
    size_t recv_len = comm_recv(comm, &conn.session_id, &msg_type, msg, sizeof(msg));
    if (recv_len == -1) {
      // timeout from recv, no message received
      switch (state) {
      case EOT_STATE_IDLE:
        printf("EOT_STATE_IDLE: waiting for user to push ARM button\n");
        printf("Press enter to simulate button press...\n");
        getchar();
        printf("Button pressed, waiting for HOT advertisement...\n");
        state = EOT_STATE_WAIT_ADV;
        break;
        // TODO: handle timeout
      default:
        break;
      }
    } else {
      /* printf("received msg type=%d len=%zu state=%d\n", msg_type.hot, recv_len, state); */
      switch (state) {
      case EOT_STATE_WAIT_ADV:
        if (msg_type.hot == HOT_MSG_ADV) {
          // received advertisement from HOT
          // generate our keypair
          if (!generate_keypair(&conn.eot_keys)) {
            fprintf(stderr, "Failed to generate EOT keypair\n");
            exit(EXIT_FAILURE);
          }
          // send our public key to HOT
          uint8_t compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          compress_pubkey(conn.eot_keys.public, compressed_pubkey);
          comm_send(comm, conn.session_id, (msg_type_t){.eot = EOT_MSG_PUBKEY}, compressed_pubkey, sizeof(compressed_pubkey));
          printf("EOT: sent public key to HOT, waiting for their pubkey and commitment...\n");
          state = EOT_STATE_KEY_EX_1;
        }
        break;
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
  enum hot_state state = HOT_STATE_IDLE;
  uint8_t msg[MAX_MSG_LEN];
  msg_type_t msg_type;
  time_t last_adv_time;
  time_t now;
  time(&last_adv_time);
  
  while (1) {
    size_t recv_len = comm_recv(comm, &conn.session_id, &msg_type, msg, sizeof(msg));
    if (recv_len == -1) {
      // timeout from recv, no message received
      switch (state) {
      case HOT_STATE_IDLE:
        printf("HOT_STATE_IDLE: waiting for user to push ARM button\n");
        printf("Press enter to simulate button press...\n");
        getchar();
        printf("Button pressed, sending advertisement...\n");
        state = HOT_STATE_ADV;
        break;
      case HOT_STATE_ADV:
        time(&now);
        if (difftime(now, last_adv_time) * 1000 >= HOT_STATE_ADV_INTERVAL_MS) {
          // send advertisement
          comm_send(comm, conn.session_id, (msg_type_t){.hot = HOT_MSG_ADV}, NULL, 0);
          last_adv_time = now;
          printf("HOT: sent advertisement\n");
        }
        break;
      default:
        break;
      }
    } else {
      switch (state) {
      case HOT_STATE_ADV:
        if (msg_type.eot == EOT_MSG_PUBKEY) {
          printf("HOT: received EOT pubkey\n");
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
