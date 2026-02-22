/**
 * Device Communication Abstraction Layer
 * Provides a platform-independent interface for device communication.
 * Current implementation uses Unix domain sockets, but can be swapped for
 * other communication mechanisms (shared memory, pipes, etc.) by implementing the
 * same interface.
 */

#ifndef COMM_H
#define COMM_H

#include <stddef.h>
#include <stdint.h>

#include "crypto.h"
#include "ext_support.h"

#ifdef TARGET_UNIX
#include <sys/types.h>
#else
typedef int ssize_t;
#endif

#define MAX_PKT_LEN 512

/**
 * Message type used in communication.
 */
typedef uint8_t msg_type_t;

/**
 * Session ID type.
 */
typedef uint32_t session_id_t;

/**
 * Unit ID type (5-digit EOT identifier).
 */
typedef uint32_t unit_id_t;

typedef enum {
    COMM_DEVICE_EOT,
    COMM_DEVICE_HOT
} comm_device_type_t;

// Opaque communication handle
typedef struct comm_handle comm_handle_t;

/**
 * Initialize device communication
 * 
 * @param device_type Whether this is an EOT or HOT device
 * @param timeout_ms Receive timeout in milliseconds
 * @return Handle to communication connection, or NULL on failure
 */
comm_handle_t* comm_init(comm_device_type_t device_type, uint32_t timeout_ms);

/**
 * Send data over communication
 * 
 * @param handle Communication handle
 * @param data Data to send
 * @param len Length of data in bytes
 * @return Number of bytes sent, or -1 on error
 */
ssize_t comm_send_raw(comm_handle_t *handle, const uint8_t *data, size_t len);

/**
 * Receive data from communication
 * 
 * @param handle Communication handle
 * @param buffer Buffer to store received data
 * @param max_len Maximum number of bytes to receive
 * @return Number of bytes received, or:
 *         -1 on timeout
 *         -2 on other error
 */
ssize_t comm_recv_raw(comm_handle_t *handle, uint8_t *buffer, size_t max_len);

/**
 * Close and cleanup communication connection
 * 
 * @param handle Communication handle
 */
void comm_close(comm_handle_t *handle);

/**
 * Communicator structure containing communication handle and timeout.
 */
typedef struct {
    uint32_t timeout_ms;
    comm_handle_t *comm_h;
} communicator_t;

/**
 * Send a message to the peer with optional signature.
 * @param comm Communicator handle
 * @param session_id Session ID
 * @param msg_type Message type
 * @param msg Message payload
 * @param msg_len Message length
 * @param shared_secret Secret for signing (NULL for no signature)
 */
void comm_send(communicator_t *comm, const session_id_t session_id, const msg_type_t msg_type,
               const uint8_t *msg, const size_t msg_len, const uint8_t *shared_secret);

/**
 * Send a legacy (old protocol) message.
 * @param comm Communicator handle
 * @param unit_id Unit ID
 * @param msg Message payload
 * @param msg_len Message length
 */
void comm_send_legacy(communicator_t *comm, const unit_id_t unit_id, const uint8_t *msg, const size_t msg_len);

/**
 * Receive a message from the peer with optional signature verification.
 * @param comm Communicator handle
 * @param session_id Pointer to store session ID
 * @param msg_type Pointer to store message type
 * @param msg Buffer for message payload
 * @param max_msg_len Maximum message length
 * @param shared_secret Secret for verification (NULL for no verification)
 * @return Length of received message, -1 on timeout, -2 on signature failure, negative for legacy messages
 */
ssize_t comm_recv(communicator_t *comm, session_id_t *session_id, msg_type_t *msg_type,
                  uint8_t *msg, const size_t max_msg_len, const uint8_t *shared_secret);

/**
 * Initialize communicator.
 * @param comm Pointer to communicator structure
 * @param device_type Device type (EOT or HOT)
 * @param timeout_ms Default timeout in milliseconds
 */
void init_communicator(communicator_t *comm, comm_device_type_t device_type, const uint32_t timeout_ms);

/**
 * Add a packet to drop for testing purposes.
 * @param pkt_num Packet number to drop (1-indexed)
 */
void add_drop_packet(int pkt_num);

#endif // COMM_H
