#ifndef DEVICES_H_INCLUDED
#define DEVICES_H_INCLUDED

#include "ext_support.h"
#include "comm.h"
#include "crypto.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEFAULT_TIMEOUT_MS 300
#define HOT_ADV_INTERVAL_MS 1000
#define PAIRING_TIMEOUT 6000
#define HOT_RETRANSMIT_INTERVAL_MS 2000
#define EOT_WAIT_ADV_TIMEOUT_MS 30000

#define PACKET_SEND_DELAY_MS_PER_BYTE 15

#define MAX_MSG_LEN 256
#define MAX_PKT_LEN 512

/**
 * Messages sent from EOT to HOT.
 */
enum eot_msg {
    EOT_MSG_PUBKEY = 0,
    EOT_MSG_NONCE,
    EOT_MSG_STATUS,
    EOT_MSG_EMERGENCY,
    EOT_MSG_UPGRADE,
};

/**
 * Messages sent from HOT to EOT.
 */
enum hot_msg {
    HOT_MSG_ADV = 0,
    HOT_MSG_PUBKEY_AND_COMMIT,
    HOT_MSG_NONCE,
    HOT_MSG_STATUS,
    HOT_MSG_EMERGENCY,
    HOT_MSG_DISCONNECT
};

/**
 * Message type used in communication.
 */
typedef uint8_t msg_type_t;

/**
 * EOT device states.
 */
enum eot_state {
    EOT_IDLE = 0,
    EOT_WAIT_ADV,
    EOT_KEY_EX_1,
    EOT_KEY_EX_2,
    EOT_PAIRED,
    EOT_LEGACY,
};

/**
 * HOT device states.
 */
enum hot_state {
    HOT_IDLE = 0,
    HOT_ADV,
    HOT_KEY_EX_1,
    HOT_WAIT_FOR_PIN,
    HOT_PAIRED,
    HOT_WAIT_FOR_STATUS,
    HOT_WAIT_FOR_EMERGENCY,
    HOT_LEGACY,
    HOT_LEGACY_ARMED,
};

/**
 * Session ID type.
 */
typedef uint32_t session_id_t;

/**
 * PIN type - 5-digit PIN.
 */
typedef uint32_t pin_t;

/**
 * Message counter for replay protection.
 */
typedef uint32_t msg_ctr_t;

/**
 * Unit ID type (5-digit EOT identifier).
 */
typedef uint32_t unit_id_t;

/**
 * EOT status data structure.
 */
typedef struct {
    uint8_t batt_cond;
    uint16_t pressure;
    uint16_t batt_charge_used;
    uint8_t valve_circuit_operational;
    uint8_t confirmation_indicator;
    uint8_t turbine_status;
    uint8_t motion_detection;
    uint8_t marker_light_battery_weak;
    uint8_t marker_light_status;
} eot_status_t;

/**
 * Communicator structure containing communication handle and timeout.
 */
typedef struct {
    uint32_t timeout_ms;
    comm_handle_t *comm_h;
} communicator_t;

/**
 * Connection information structure containing session keys and state.
 */
typedef struct {
    session_id_t session_id;
    keypair_t eot_keys;
    keypair_t hot_keys;
    uint8_t shared_secret[SHARED_SECRET_SIZE];
    commitment_t hot_commitment;
    nonce_t eot_nonce;
    nonce_t hot_nonce;
    pin_t pin;
    msg_ctr_t ctr;
} conn_info_t;

/**
 * Protocol timer type (alias for ext_timer_t).
 */
typedef ext_timer_t protocol_timer_t;

/**
 * Get current time for protocol timing.
 * @param t Pointer to timer struct to fill with current time
 */
static inline void timer_now(protocol_timer_t *t) {
    ext_timer_now(t);
}

/**
 * Calculate time difference in milliseconds.
 * @param end The later time
 * @param start The earlier time
 * @return Difference in milliseconds
 */
static inline int timer_diff_ms(const protocol_timer_t *end, const protocol_timer_t *start) {
    return ext_timer_diff_ms(end, start);
}

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
 * Get current EOT status data (dummy implementation).
 * @param status Pointer to status structure to fill
 */
void get_eot_status(eot_status_t *status);

/**
 * Display EOT status to console.
 * @param status Status structure to display
 */
void display_eot_status(eot_status_t *status);

/**
 * Trigger emergency brake on EOT.
 */
void eot_emergency_brake(void);

/**
 * Wait for ARM button press (simulated with stdin).
 */
void wait_for_arm_button_press(void);

/**
 * Handle legacy message received by EOT.
 * @param msg Message payload
 * @param msg_len Message length
 */
void eot_handle_legacy_message(const uint8_t *msg, size_t msg_len);

/**
 * Handle legacy message received by HOT.
 * @param msg Message payload
 * @param msg_len Message length
 */
void hot_handle_legacy_message(const uint8_t *msg, size_t msg_len);

/**
 * Main EOT state machine loop.
 * @param comm Communicator handle
 * @param unit_id EOT unit ID
 */
void eot_run(communicator_t *comm, unit_id_t unit_id);

/**
 * Main HOT state machine loop.
 * @param comm Communicator handle
 */
void hot_run(communicator_t *comm);

/**
 * Initialize communicator.
 * @param comm Pointer to communicator structure
 * @param device_type Device type (EOT or HOT)
 * @param timeout_ms Default timeout in milliseconds
 */
void init_communicator(communicator_t *comm, comm_device_type_t device_type, const uint32_t timeout_ms);

/**
 * EOT device main entry point.
 * @return 0 on exit
 */
int eot_main(void);

/**
 * HOT device main entry point.
 * @return 0 on exit
 */
int hot_main(void);

/**
 * Add a packet to drop for testing purposes.
 * @param pkt_num Packet number to drop (1-indexed)
 */
void add_drop_packet(int pkt_num);

#ifdef __cplusplus
}
#endif

#endif
