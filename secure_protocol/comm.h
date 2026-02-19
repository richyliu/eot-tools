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
#include <sys/types.h>

// Opaque communication handle
typedef struct comm_handle comm_handle_t;

/**
 * Initialize device communication
 * 
 * @param send_path Path/identifier for sending (implementation-specific)
 * @param recv_path Path/identifier for receiving (implementation-specific)
 * @param timeout_ms Receive timeout in milliseconds
 * @return Handle to communication connection, or NULL on failure
 */
comm_handle_t* comm_init(const char *send_path, const char *recv_path, uint32_t timeout_ms);

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

#endif // COMM_H
