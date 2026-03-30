/**
 * This is an upgraded protocol for communication between end-of-train
 * devices (EOTD) and head-of-train devices (HOTD) that supports
 * message authentication using ECDH and HMAC-SHA256, as well as
 * a legacy mode for backward compatibility.
 */

#include <stdint.h>

#include "comm.h"
#include "crypto.h"
#include "devices.h"
#include "ext_support.h"
#include "profiling.h"

static int g_legacy_only = 0;

void set_legacy_only(int legacy_only) { g_legacy_only = legacy_only; }

int is_legacy_only(void) { return g_legacy_only; }

void get_eot_status(eot_status_t *status) {
  status->batt_cond = 2;
  status->pressure = 100;
  status->batt_charge_used = 50;
  status->valve_circuit_operational = 1;
  status->confirmation_indicator = 0;
  status->turbine_status = 1;
  status->motion_detection = 0;
  status->marker_light_battery_weak = 0;
  status->marker_light_status = 1;
}

void display_eot_status(eot_status_t *status) {
  ext_io_puts("EOT Status:\n");
  ext_io_printf("  Battery Condition: %u\n", status->batt_cond);
  ext_io_printf("  Pressure: %u\n", status->pressure);
  ext_io_printf("  Battery Charge Used: %u\n", status->batt_charge_used);
  ext_io_printf("  Valve Circuit Operational: %u\n",
                status->valve_circuit_operational);
  ext_io_printf("  Confirmation Indicator: %u\n",
                status->confirmation_indicator);
  ext_io_printf("  Turbine Status: %u\n", status->turbine_status);
  ext_io_printf("  Motion Detection: %u\n", status->motion_detection);
  ext_io_printf("  Marker Light Battery Weak: %u\n",
                status->marker_light_battery_weak);
  ext_io_printf("  Marker Light Status: %u\n", status->marker_light_status);
  ext_io_puts("\n");
}

void eot_emergency_brake(void) {
  ext_io_puts("EOT: Emergency brake activated!\n");
}

void wait_for_arm_button_press(void) {
  ext_io_clear_input();
  ext_io_puts("Press enter to simulate ARM button press...\n");
  ext_io_getc();
}

void eot_handle_legacy_message(const uint8_t *msg, size_t msg_len) {
  (void)msg;
  (void)msg_len;
  ext_io_puts("EOT: Received legacy message (not implemented)\n");
}

void hot_handle_legacy_message(const uint8_t *msg, size_t msg_len) {
  (void)msg;
  (void)msg_len;
  ext_io_puts("HOT: Received legacy message (not implemented)\n");
}

void eot_run(communicator_t *comm, unit_id_t unit_id) {
  conn_info_t conn;
  ext_memset(&conn, 0, sizeof(conn));
  session_id_t recved_session_id = 0;
  enum eot_state state = EOT_IDLE;
  uint8_t msg[MAX_MSG_LEN];
  msg_type_t msg_type;
  eot_status_t status;
  protocol_timer_t adv_start;
  protocol_timer_t pairing_start;
  protocol_timer_t last_status_time;
  protocol_timer_t now;
  int next_status_delay_ms = 30000 + (ext_random_u32() % 5000);
  timer_now(&adv_start);
  timer_now(&pairing_start);
  timer_now(&last_status_time);
  timer_now(&now);
  uint8_t *shared_secret;
  int choice = 0;

  while (1) {
    log_stack_usage();
    if (state == EOT_PAIRED) {
      shared_secret = conn.shared_secret;
    } else {
      shared_secret = NULL;
    }
    ssize_t recv_len = comm_recv(comm, &recved_session_id, &msg_type, msg,
                                 sizeof(msg), shared_secret);
    timer_now(&now);
    if (recv_len == -1) {
      switch (state) {
      case EOT_IDLE:
        if (is_legacy_only()) {
          ext_io_puts("Legacy-only mode: Entering legacy mode and sending legacy ARM command\n");
          state = EOT_LEGACY;
          timer_now(&last_status_time);
          comm_send_legacy(comm, unit_id, (uint8_t *)"ARM", 3);
          break;
        }
        ext_io_puts("EOT_IDLE: waiting for user to push TEST button\n");
        ext_io_puts("Options:\n");
        ext_io_puts("  1: Push TEST button to start pairing\n");
        ext_io_puts("  2: Hold TEST button for 5 seconds to enter legacy mode\n");
        ext_io_puts("Enter choice: ");
        ext_io_flush();
        ext_io_scan_int(&choice);
        if (choice == 1) {
          ext_io_puts("Button pressed, waiting for HOT advertisement...\n");
          timer_now(&adv_start);
          state = EOT_WAIT_ADV;
        } else if (choice == 2) {
          ext_io_printf("EOT entering legacy mode. Unit ID: %05u\n", unit_id);
          state = EOT_LEGACY;
          timer_now(&last_status_time);
          comm_send_legacy(comm, unit_id, (uint8_t *)"ARM", 3);
          comm_send(comm, 0, (msg_type_t)EOT_MSG_UPGRADE, (uint8_t *)&unit_id, sizeof(unit_id_t), NULL);
        } else {
          ext_io_puts("Invalid choice, staying in idle state.\n");
        }
        break;
      case EOT_WAIT_ADV:
        if (timer_diff_ms(&now, &adv_start) >= EOT_WAIT_ADV_TIMEOUT_MS) {
          ext_io_puts(
              "EOT: Waiting for HOT timed out. Returning to idle state.\n");
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
      case EOT_PAIRED: {
        ext_io_set_nonblocking(1);
        char buf[16];
        if (ext_io_getline(buf, sizeof(buf)) >= 0) {
          ext_io_puts("Pressed enter (ARM button), disconnecting and searching "
                      "for new HOT...\n");
          timer_now(&adv_start);
          state = EOT_WAIT_ADV;
          ext_memset(&conn, 0, sizeof(conn));
        }
        ext_io_set_nonblocking(0);
        break;
      }
      case EOT_LEGACY: {
        if (timer_diff_ms(&now, &last_status_time) >= next_status_delay_ms) {
          ext_io_puts("EOT: Sending periodic legacy status update\n");
          get_eot_status(&status);
          comm_send_legacy(comm, unit_id, (uint8_t *)&status, sizeof(status));
          if (!is_legacy_only()) {
            comm_send(comm, 0, (msg_type_t)EOT_MSG_UPGRADE, (uint8_t *)&unit_id, sizeof(unit_id_t), NULL);
          }
          last_status_time = now;
          next_status_delay_ms = 30000 + (ext_random_u32() % 5000);
        }
        ext_io_set_nonblocking(1);
        char buf[16];
        if (ext_io_getline(buf, sizeof(buf)) >= 0) {
          ext_io_puts("EOT: TEST button pressed, sending legacy status update\n");
          get_eot_status(&status);
          comm_send_legacy(comm, unit_id, (uint8_t *)&status, sizeof(status));
          if (!is_legacy_only()) {
            comm_send(comm, 0, (msg_type_t)EOT_MSG_UPGRADE, (uint8_t *)&unit_id, sizeof(unit_id_t), NULL);
          }
          last_status_time = now;
        }
        ext_io_set_nonblocking(0);
        break;
      }
      default:
        break;
      }
    } else if (recv_len == -2) {
      ext_io_puts("EOT: received message with invalid signature, ignoring.\n");
    } else if (recv_len < 0) {
      if (state != EOT_LEGACY) {
        ext_io_puts("EOT: received legacy message while not in legacy state, "
                    "ignoring.\n");
        continue;
      }
      if (-recv_len < (ssize_t)sizeof(unit_id_t)) {
        ext_io_eprintf(
            "EOT: received legacy message too short (%zd bytes), ignoring.\n",
            -recv_len);
        continue;
      }
      unit_id_t recved_legacy_unit_id;
      ext_memcpy(&recved_legacy_unit_id, msg, sizeof(unit_id_t));
      if (recved_legacy_unit_id != unit_id) {
          ext_io_printf("EOT: received legacy message for unit %u, but I am unit %u. Ignoring.\n", recved_legacy_unit_id, unit_id);
          continue;
      }
      
      const uint8_t *payload = msg + sizeof(unit_id_t);
      size_t payload_len = (size_t)(-recv_len) - sizeof(unit_id_t);
      
      if (payload_len >= 4 && ext_memcmp(payload, "STAT", 4) == 0) {
        ext_io_puts("EOT: Received legacy status request\n");
        get_eot_status(&status);
        comm_send_legacy(comm, unit_id, (uint8_t *)&status, sizeof(status));
      } else if (payload_len >= 2 && ext_memcmp(payload, "EB", 2) == 0) {
        ext_io_puts("EOT: Received legacy emergency brake request\n");
        eot_emergency_brake();
        comm_send_legacy(comm, unit_id, (uint8_t *)"ACK EB", 6);
      } else {
        eot_handle_legacy_message(payload, payload_len);
      }
    } else {
      switch (state) {
      case EOT_WAIT_ADV:
        if (msg_type == HOT_MSG_ADV) {
          ext_io_puts("EOT: received advertisement from HOT, initiating "
                      "connection...\n");
          timer_now(&pairing_start);
          ext_memset(&conn, 0, sizeof(conn));
          ext_io_printf("EOT: establishing connection with ID %u\n",
                        recved_session_id);
          conn.session_id = recved_session_id;

          profile_start("EOT_KEYGEN");
          int keygen_ok = generate_keypair(&conn.eot_keys);
          profile_end("EOT_KEYGEN");
          if (!keygen_ok) {
            ext_io_eprintf("Failed to generate EOT keypair\n");
            ext_exit(1);
          }
          uint8_t compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          profile_start("EOT_COMPRESS_PUBKEY");
          compress_pubkey(conn.eot_keys.public_key, compressed_pubkey);
          profile_end("EOT_COMPRESS_PUBKEY");
          comm_send(comm, conn.session_id, (msg_type_t)EOT_MSG_PUBKEY,
                    compressed_pubkey, sizeof(compressed_pubkey), NULL);
          ext_io_puts("EOT: sent public key to HOT, waiting for their pubkey "
                      "and commitment...\n");
          state = EOT_KEY_EX_1;
        }
        break;
      case EOT_KEY_EX_1:
        if (msg_type == HOT_MSG_PUBKEY_AND_COMMIT &&
            recved_session_id == conn.session_id &&
            recv_len == (ssize_t)(COMPRESSED_PUBKEY_SIZE + COMMITMENT_SIZE)) {
          uint8_t hot_compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          ext_memcpy(hot_compressed_pubkey, msg, COMPRESSED_PUBKEY_SIZE);
          ext_memcpy(conn.hot_commitment.data, msg + COMPRESSED_PUBKEY_SIZE,
                     COMMITMENT_SIZE);
          profile_start("EOT_DECOMPRESS_PUBKEY");
          int valid = decompress_pubkey(hot_compressed_pubkey,
                                        conn.hot_keys.public_key);
          profile_end("EOT_DECOMPRESS_PUBKEY");
          if (!valid) {
            ext_io_puts(
                "EOT: Invalid HOT public key received, aborting connection.\n");
            state = EOT_IDLE;
            break;
          }
          ext_io_puts(
              "EOT: received HOT pubkey and commitment, generating nonce...\n");
          generate_nonce(&conn.eot_nonce);
          comm_send(comm, conn.session_id, (msg_type_t)EOT_MSG_NONCE,
                    conn.eot_nonce.data, sizeof(conn.eot_nonce.data), NULL);
          ext_io_puts("EOT: sent nonce to HOT, waiting for their nonce...\n");
          state = EOT_KEY_EX_2;
        }
        break;
      case EOT_KEY_EX_2:
        if (msg_type == HOT_MSG_NONCE && recved_session_id == conn.session_id &&
            recv_len == (ssize_t)NONCE_SIZE) {
          ext_memcpy(conn.hot_nonce.data, msg, NONCE_SIZE);
          profile_start("EOT_VERIFY_COMMITMENT");
          int verified =
              verify_commitment(&conn.hot_nonce, &conn.hot_commitment);
          profile_end("EOT_VERIFY_COMMITMENT");
          if (!verified) {
            ext_io_eprintf("EOT: Commitment verification failed! Aborting.\n");
            state = EOT_IDLE;
            break;
          }
          timer_now(&now);
          ext_io_printf("EOT: pairing took %d ms\n",
                        timer_diff_ms(&now, &pairing_start));
          profile_start("EOT_COMPUTE_PIN");
          conn.pin =
              compute_pin(conn.eot_keys.public_key, conn.hot_keys.public_key,
                          &conn.eot_nonce, &conn.hot_nonce);
          profile_end("EOT_COMPUTE_PIN");
          profile_start("EOT_COMPUTE_SHARED");
          compute_shared_secret(conn.eot_keys.private_key,
                                conn.hot_keys.public_key, conn.shared_secret);
          profile_end("EOT_COMPUTE_SHARED");
          ext_io_printf(
              "EOT: received HOT nonce and verified commitment. PIN is %05u\n",
              conn.pin);
          ext_io_puts("Please enter the PIN in the HOT and press the TEST "
                      "button once confirmed.\n");
          wait_for_arm_button_press();
          ext_io_puts("Pairing successful! Press enter at any time to "
                      "disconnect and return to waiting for advertisement.\n");
          state = EOT_PAIRED;
        }
        break;
      case EOT_PAIRED:
        if (recved_session_id == conn.session_id) {
          if (recv_len < (ssize_t)sizeof(conn.ctr)) {
            ext_io_puts(
                "EOT: received message too short for counter, ignoring.\n");
            break;
          }
          msg_ctr_t recv_ctr = *(msg_ctr_t *)msg;
          if (recv_ctr <= conn.ctr) {
            ext_io_printf("EOT: received message with old counter (%u <= %u), "
                          "ignoring.\n",
                          recv_ctr, conn.ctr);
            break;
          }
          conn.ctr = recv_ctr;
          if (msg_type == HOT_MSG_STATUS) {
            get_eot_status(&status);
            comm_send(comm, conn.session_id, (msg_type_t)EOT_MSG_STATUS,
                      (uint8_t *)&status, sizeof(status), conn.shared_secret);
            ext_io_puts("EOT: sent status update to HOT.\n");
          } else if (msg_type == HOT_MSG_EMERGENCY) {
            eot_emergency_brake();
            comm_send(comm, conn.session_id, (msg_type_t)EOT_MSG_EMERGENCY,
                      NULL, 0, conn.shared_secret);
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
  conn_info_t conn;
  ext_memset(&conn, 0, sizeof(conn));
  session_id_t recved_session_id = 0;
  enum hot_state state = HOT_IDLE;
  uint8_t msg[MAX_MSG_LEN];
  msg_type_t msg_type;
  protocol_timer_t last_adv_time;
  protocol_timer_t last_transmit_time;
  protocol_timer_t pairing_start;
  protocol_timer_t legacy_arm_timer;
  protocol_timer_t now;
  timer_now(&last_adv_time);
  timer_now(&last_transmit_time);
  timer_now(&pairing_start);
  timer_now(&legacy_arm_timer);
  timer_now(&now);
  uint32_t input_pin = 0;
  int choice;
  uint8_t *shared_secret;
  unit_id_t legacy_unit_id = 0;
  unit_id_t recent_upgradable_legacy_unit_id = 0;

  if (is_legacy_only()) {
    ext_io_puts("Legacy-only mode: waiting for legacy ARM command...\n");
    ext_io_puts("Enter 5-digit unit ID to pair with: ");
    ext_io_flush();
    while (ext_io_scan_int(&choice) != 0 || choice < 1 || choice > 99999) {
      ext_io_puts("Invalid ID, enter 5-digit unit ID: ");
      ext_io_flush();
    }
    legacy_unit_id = (unit_id_t)choice;
    state = HOT_LEGACY;
    ext_io_printf("HOT: Legacy only mode active for unit ID %05u\n", legacy_unit_id);
  }

  while (1) {
    log_stack_usage();
    if (state == HOT_PAIRED || state == HOT_WAIT_FOR_STATUS ||
        state == HOT_WAIT_FOR_EMERGENCY) {
      shared_secret = conn.shared_secret;
    } else {
      shared_secret = NULL;
    }
    ssize_t recv_len = comm_recv(comm, &recved_session_id, &msg_type, msg,
                                 sizeof(msg), shared_secret);
    timer_now(&now);
    if (recv_len == -1) {
      switch (state) {
      case HOT_IDLE:
        ext_io_puts("HOT_IDLE: waiting for user to push ARM button\n");
        ext_io_puts("Options:\n");
        ext_io_puts("  -1: Push ARM button to start pairing\n");
        ext_io_puts(
            "  <5-digit unit ID>: Enter legacy mode with given unit ID\n");
        ext_io_puts("Enter choice: ");
        ext_io_flush();
        ext_io_scan_int(&choice);
        if (choice == -1) {
          ext_io_puts("Button pressed, sending advertisement...\n");
          ext_memset(&conn, 0, sizeof(conn));
          conn.session_id = generate_session_id();
          ext_io_printf("session id: %u\n", conn.session_id);
          state = HOT_ADV;
        } else if (choice >= 1 && choice <= 99999) {
          if (recent_upgradable_legacy_unit_id == (unit_id_t)choice) {
            ext_io_printf("Requested to pair in legacy mode with unit ID %05u, "
                          "but it supports the new protocol. Please use the "
                          "new protocol to pair.\n",
                          choice);
          } else {
            ext_io_printf("Entering legacy mode with unit ID %05u. Waiting for EOT message...\n", choice);
            legacy_unit_id = (unit_id_t)choice;
            state = HOT_LEGACY;
            timer_now(&pairing_start);
          }
        } else {
          ext_io_puts("Invalid choice, staying in idle state.\n");
        }
        break;
      case HOT_LEGACY: {
        ext_io_set_nonblocking(1);
        char buf[16];
        if (ext_io_getline(buf, sizeof(buf)) >= 0) {
            if (timer_diff_ms(&now, &legacy_arm_timer) < 5000) {
                ext_io_puts("HOT: ARM NOW pressed! SYSTEM ARMED (Legacy Mode)\n");
                ext_io_printf("HOT: legacy pairing took %d ms\n", timer_diff_ms(&now, &pairing_start));
                state = HOT_LEGACY_ARMED;
            } else {
                ext_io_puts("HOT: ARM timer expired or button pressed too early. Wait for EOT status.\n");
            }
        }
        ext_io_set_nonblocking(0);
        break;
      }
      case HOT_LEGACY_ARMED: {
        ext_io_puts("Legacy ARMED Options:\n");
        ext_io_puts("  1. Request legacy status update\n");
        ext_io_puts("  2. Legacy emergency brake\n");
        ext_io_puts("  3. Disconnect\n");
        ext_io_puts("Enter choice: ");
        ext_io_flush();
        choice = 0;
        ext_io_scan_int(&choice);
        if (choice == 1) {
          ext_io_puts("HOT: Requesting legacy status update\n");
          timer_now(&last_transmit_time);
          comm_send_legacy(comm, legacy_unit_id, (uint8_t *)"STAT", 4);
        } else if (choice == 2) {
          ext_io_puts("HOT: Sending legacy emergency brake request\n");
          comm_send_legacy(comm, legacy_unit_id, (uint8_t *)"EB", 2);
          state = HOT_WAIT_FOR_LEGACY_EB_ACK;
          timer_now(&last_transmit_time);
        } else if (choice == 3) {
          ext_io_puts("HOT: Disconnecting legacy mode\n");
          state = HOT_IDLE;
        }
        break;
      }
      case HOT_WAIT_FOR_LEGACY_EB_ACK:
        if (timer_diff_ms(&now, &last_transmit_time) >= 10000) {
          ext_io_puts("HOT: Retransmitting legacy emergency brake request\n");
          comm_send_legacy(comm, legacy_unit_id, (uint8_t *)"EB", 2);
          timer_now(&last_transmit_time);
        }
        break;
      case HOT_ADV:
        if (timer_diff_ms(&now, &last_adv_time) >= HOT_ADV_INTERVAL_MS) {
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_ADV, NULL, 0,
                    NULL);
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
        ext_io_printf(
            "HOT: waiting for user to input PIN (expected PIN is %05u)\n",
            conn.pin);
        for (int i = 0; i < 3; i++) {
          ext_io_printf("Enter PIN (attempt %d of 3): ", i + 1);
          ext_io_flush();
          ext_io_scan_uint(&input_pin);
          ext_io_printf("You entered: %05u\n", input_pin);
          if (input_pin == conn.pin) {
            ext_io_puts("PIN correct! Pairing successful. Press the ARM button "
                        "on the EOT to confirm.\n");
            state = HOT_PAIRED;
            break;
          } else {
            ext_io_puts("Incorrect PIN. Try again.\n");
          }
        }
        if (state != HOT_PAIRED) {
          ext_io_puts(
              "Failed to enter correct PIN. Returning to idle state.\n");
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

        conn.ctr++;
        if (choice == 1) {
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_STATUS,
                    (uint8_t *)&conn.ctr, sizeof(conn.ctr), conn.shared_secret);
          ext_io_puts("HOT: sent status update request\n");
          state = HOT_WAIT_FOR_STATUS;
        } else if (choice == 2) {
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_EMERGENCY,
                    (uint8_t *)&conn.ctr, sizeof(conn.ctr), conn.shared_secret);
          ext_io_puts("HOT: sent emergency brake request\n");
          state = HOT_WAIT_FOR_EMERGENCY;
        } else if (choice == 3) {
          ext_io_puts("HOT: Disconnecting...\n");
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_DISCONNECT,
                    (uint8_t *)&conn.ctr, sizeof(conn.ctr), conn.shared_secret);
          state = HOT_IDLE;
          ext_memset(&conn, 0, sizeof(conn));
        } else {
          ext_io_puts("Invalid choice.\n");
        }
        timer_now(&last_transmit_time);
        break;
      case HOT_WAIT_FOR_STATUS:
      case HOT_WAIT_FOR_EMERGENCY:
        conn.ctr++;
        if (timer_diff_ms(&now, &last_transmit_time) >=
            HOT_RETRANSMIT_INTERVAL_MS) {
          if (state == HOT_WAIT_FOR_STATUS) {
            comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_STATUS,
                      (uint8_t *)&conn.ctr, sizeof(conn.ctr),
                      conn.shared_secret);
            ext_io_puts("HOT: retransmitted status update request\n");
          } else if (state == HOT_WAIT_FOR_EMERGENCY) {
            comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_EMERGENCY,
                      (uint8_t *)&conn.ctr, sizeof(conn.ctr),
                      conn.shared_secret);
            ext_io_puts("HOT: retransmitted emergency brake request\n");
          }
          last_transmit_time = now;
        }
        break;
      default:
        break;
      }
    } else if (recv_len == -2) {
      ext_io_puts("HOT: received message with invalid signature, ignoring.\n");
    } else if (recv_len < 0) {
      unit_id_t recved_legacy_unit_id;
      ext_memcpy(&recved_legacy_unit_id, msg, sizeof(unit_id_t));
      const uint8_t *payload = msg + sizeof(unit_id_t);
      size_t payload_len = (size_t)(-recv_len) - sizeof(unit_id_t);

      if (state == HOT_LEGACY) {
        if (recved_legacy_unit_id == legacy_unit_id) {
          ext_io_printf("HOT: Received status from EOT %05u. Press enter to ARM NOW (you have 5 seconds)...\n", legacy_unit_id);
          timer_now(&legacy_arm_timer);
        }
      } else if (state == HOT_LEGACY_ARMED || state == HOT_WAIT_FOR_LEGACY_EB_ACK) {
        if (recved_legacy_unit_id == legacy_unit_id) {
          if (payload_len == sizeof(eot_status_t)) {
             eot_status_t status;
             ext_memcpy(&status, payload, sizeof(eot_status_t));
             display_eot_status(&status);
             if (timer_diff_ms(&now, &last_transmit_time) < 10000) {
                 ext_io_printf("HOT: legacy status update took %d ms\n", timer_diff_ms(&now, &last_transmit_time));
             } else {
                 ext_io_puts("HOT: received unsolicited legacy status update\n");
             }
          } else if (payload_len >= 6 && ext_memcmp(payload, "ACK EB", 6) == 0) {
             ext_io_printf("HOT: Received legacy emergency brake acknowledgment. %d ms elapsed since last request.\n", timer_diff_ms(&now, &last_transmit_time));
             state = HOT_LEGACY_ARMED;
          }
        }
      } else {
        if (!is_legacy_only()) {
          ext_io_printf("HOT: modern mode active, ignoring legacy ARM command "
                        "from unit ID %u to prevent downgrade attack.\n",
                        recved_legacy_unit_id);
        } else if (payload_len >= 3 && ext_memcmp(payload, "ARM", 3) == 0) {
          ext_io_printf("HOT: received legacy ARM command from unit ID %u, "
                        "entering legacy ARMED mode.\n",
                        recved_legacy_unit_id);
          legacy_unit_id = recved_legacy_unit_id;
          state = HOT_LEGACY_ARMED;
        } else {
          hot_handle_legacy_message(payload, payload_len);
        }
      }
    } else {
      if (msg_type == EOT_MSG_UPGRADE &&
          recv_len == (ssize_t)sizeof(unit_id_t)) {
        ext_memcpy(&recent_upgradable_legacy_unit_id, msg, sizeof(unit_id_t));
        ext_io_printf(
            "HOT: received protocol upgrade request from legacy unit ID %u\n",
            recent_upgradable_legacy_unit_id);
        if (!is_legacy_only() && (state == HOT_LEGACY || state == HOT_LEGACY_ARMED || state == HOT_WAIT_FOR_LEGACY_EB_ACK) &&
            recent_upgradable_legacy_unit_id == legacy_unit_id) {
          ext_io_printf("HOT: Requested to pair in legacy mode with unit ID %05u, "
                        "but it supports the new protocol. Aborting legacy pairing to prevent downgrade attack.\n", legacy_unit_id);
          state = HOT_IDLE;
        }
      }
      switch (state) {
      case HOT_ADV:
        if (msg_type == EOT_MSG_PUBKEY &&
            recved_session_id == conn.session_id) {
          ext_io_puts("HOT: received EOT pubkey, initiating connection...\n");
          timer_now(&pairing_start);

          uint8_t eot_compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          ext_memcpy(eot_compressed_pubkey, msg, COMPRESSED_PUBKEY_SIZE);
          profile_start("HOT_DECOMPRESS_PUBKEY");
          decompress_pubkey(eot_compressed_pubkey, conn.eot_keys.public_key);
          profile_end("HOT_DECOMPRESS_PUBKEY");

          profile_start("HOT_KEYGEN");
          int keygen_ok = generate_keypair(&conn.hot_keys);
          profile_end("HOT_KEYGEN");
          if (!keygen_ok) {
            ext_io_eprintf("Failed to generate HOT keypair\n");
            ext_exit(1);
          }
          generate_nonce(&conn.hot_nonce);
          profile_start("HOT_CREATE_COMMITMENT");
          create_commitment(&conn.hot_nonce, &conn.hot_commitment);
          profile_end("HOT_CREATE_COMMITMENT");
          uint8_t compressed_pubkey[COMPRESSED_PUBKEY_SIZE];
          profile_start("HOT_COMPRESS_PUBKEY");
          compress_pubkey(conn.hot_keys.public_key, compressed_pubkey);
          profile_end("HOT_COMPRESS_PUBKEY");
          uint8_t payload[COMPRESSED_PUBKEY_SIZE + COMMITMENT_SIZE];
          ext_memcpy(payload, compressed_pubkey, COMPRESSED_PUBKEY_SIZE);
          ext_memcpy(payload + COMPRESSED_PUBKEY_SIZE, conn.hot_commitment.data,
                     COMMITMENT_SIZE);
          comm_send(comm, conn.session_id,
                    (msg_type_t)HOT_MSG_PUBKEY_AND_COMMIT, payload,
                    sizeof(payload), NULL);

          ext_io_puts("HOT: sent pubkey and commitment to EOT, waiting for "
                      "their nonce...\n");
          state = HOT_KEY_EX_1;
        }
        break;
      case HOT_KEY_EX_1:
        if (msg_type == EOT_MSG_NONCE && recved_session_id == conn.session_id &&
            recv_len == (ssize_t)NONCE_SIZE) {
          ext_io_puts("HOT: received EOT nonce, sending our nonce...\n");
          ext_memcpy(conn.eot_nonce.data, msg, NONCE_SIZE);
          profile_start("HOT_COMPUTE_PIN");
          conn.pin =
              compute_pin(conn.eot_keys.public_key, conn.hot_keys.public_key,
                          &conn.eot_nonce, &conn.hot_nonce);
          profile_end("HOT_COMPUTE_PIN");
          profile_start("HOT_COMPUTE_SHARED");
          compute_shared_secret(conn.hot_keys.private_key,
                                conn.eot_keys.public_key, conn.shared_secret);
          profile_end("HOT_COMPUTE_SHARED");
          comm_send(comm, conn.session_id, (msg_type_t)HOT_MSG_NONCE,
                    conn.hot_nonce.data, sizeof(conn.hot_nonce.data), NULL);
          ext_io_puts(
              "HOT: sent nonce to EOT, waiting for user to input PIN...\n");
          state = HOT_WAIT_FOR_PIN;
        }
      case HOT_WAIT_FOR_STATUS:
        if (msg_type == EOT_MSG_STATUS &&
            recved_session_id == conn.session_id &&
            recv_len == (ssize_t)sizeof(eot_status_t)) {
          eot_status_t status;
          ext_memcpy(&status, msg, sizeof(eot_status_t));
          display_eot_status(&status);
          ext_io_printf("HOT: status update took %d ms\n",
                        timer_diff_ms(&now, &last_transmit_time));
          state = HOT_PAIRED;
        }
        break;
      case HOT_WAIT_FOR_EMERGENCY:
        if (msg_type == EOT_MSG_EMERGENCY &&
            recved_session_id == conn.session_id) {
          ext_io_printf("HOT: received emergency brake confirmation from EOT. "
                        "%d ms elapsed since last request.\n",
                        timer_diff_ms(&now, &last_transmit_time));
          state = HOT_PAIRED;
        }
        break;
      default:
        break;
      }
    }
  }
}

int eot_main(const char *socket_path1, const char *socket_path2) {
  communicator_t comm;
  unit_id_t sample_unit_id = 12345;

  ext_timer_init_cycles();
  ext_io_puts("EOT starting...\n");
  init_communicator(&comm, COMM_DEVICE_EOT, DEFAULT_TIMEOUT_MS, socket_path1,
                    socket_path2);
  eot_run(&comm, sample_unit_id);
  return 0;
}

int hot_main(const char *socket_path1, const char *socket_path2) {
  communicator_t comm;

  ext_timer_init_cycles();
  ext_io_puts("HOT starting...\n");
  init_communicator(&comm, COMM_DEVICE_HOT, DEFAULT_TIMEOUT_MS, socket_path1,
                    socket_path2);
  hot_run(&comm);
  return 0;
}
