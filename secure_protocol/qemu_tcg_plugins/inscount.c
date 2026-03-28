/*
 * License: GNU GPL, version 2 or later.
 *
 * QEMU TCG plugin: instruction counter + magic-address section profiler.
 *
 * Guest writes a string to PROFILE_MAGIC_NAME (four 4-byte words to store the
 * name) and then a 1 to PROFILE_MAGIC_CONTROL to begin profiling. Once the
 * guest is done, it writes 2 to PROFILE_MAGIC_CONTROL. The plugin will output
 * the number of instructions executed between the two writes.
 */
#include "inscount.h"
#include <glib.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

static qemu_plugin_u64 insn_count;

typedef struct {
  char name[PROFILE_MAGIC_NAME_SIZE];
  uint64_t start_count;
} SectionState;

static struct qemu_plugin_scoreboard *section_states;

/* ------------------------------------------------------------------ */
/* Instruction counting                                                 */
/* ------------------------------------------------------------------ */

static void vcpu_insn_exec_before(unsigned int cpu_index, void *udata) {
  qemu_plugin_u64_add(insn_count, cpu_index, 1);
}

/* ------------------------------------------------------------------ */
/* Magic-address memory callback                                        */
/* ------------------------------------------------------------------ */

static void vcpu_mem_cb(unsigned int cpu_index, qemu_plugin_meminfo_t info,
                        uint64_t vaddr, void *udata) {
  if (vaddr < PROFILE_MAGIC_BASE || vaddr >= PROFILE_MAGIC_END) {
    return;
  }

  qemu_plugin_mem_value val = qemu_plugin_mem_get_value(info);
  uint64_t guest_val = 0;
  int guest_val_size = 4;
  switch (val.type) {
  case QEMU_PLUGIN_MEM_VALUE_U8:
    guest_val = val.data.u8;
    guest_val_size = 1;
    break;
  case QEMU_PLUGIN_MEM_VALUE_U16:
    guest_val = val.data.u16;
    guest_val_size = 2;
    break;
  case QEMU_PLUGIN_MEM_VALUE_U32:
    guest_val = val.data.u32;
    guest_val_size = 4;
    break;
  case QEMU_PLUGIN_MEM_VALUE_U64:
    guest_val = val.data.u64;
    guest_val_size = 8;
    break;
  default:
    return;
  }

  SectionState *state = qemu_plugin_scoreboard_find(section_states, cpu_index);

  if (vaddr == PROFILE_MAGIC_CONTROL) {
    if (guest_val == 1) {
      state->start_count = qemu_plugin_u64_get(insn_count, cpu_index);
    } else if (guest_val == 2) {
      uint64_t end_count = qemu_plugin_u64_get(insn_count, cpu_index);
      uint64_t delta = end_count - state->start_count;
      g_autoptr(GString) out = g_string_new(NULL);
      g_string_append_printf(out, "[PROFILE] %s: %" PRIu64 " instructions\n",
                             state->name, delta);
      qemu_plugin_outs(out->str);
    }
  } else if (vaddr >= PROFILE_MAGIC_NAME_START &&
             vaddr < PROFILE_MAGIC_NAME_END) {
    size_t offset = vaddr - PROFILE_MAGIC_NAME_START;
    for (int i = 0; i < guest_val_size; i++) {
      state->name[offset + i] = (guest_val >> (i * 8)) & 0xFF;
    }
    // make sure name is always null terminated
    state->name[PROFILE_MAGIC_NAME_SIZE - 1] = '\0';
  } else {
    g_autoptr(GString) out = g_string_new(NULL);
    g_string_append_printf(out, "[inscount] unknown magic address: %llx\n",
                           vaddr);
    qemu_plugin_outs(out->str);
  }
}

/* ------------------------------------------------------------------ */
/* Translation-block callback: register per-insn hooks                 */
/* ------------------------------------------------------------------ */

static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
  size_t n = qemu_plugin_tb_n_insns(tb);

  for (size_t i = 0; i < n; i++) {
    struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);

    qemu_plugin_register_vcpu_insn_exec_cb(insn, vcpu_insn_exec_before,
                                           QEMU_PLUGIN_CB_NO_REGS, NULL);

    qemu_plugin_register_vcpu_mem_cb(insn, vcpu_mem_cb, QEMU_PLUGIN_CB_NO_REGS,
                                     QEMU_PLUGIN_MEM_W, NULL);
  }
}

/* ------------------------------------------------------------------ */
/* Exit: print per-cpu and total instruction counts                     */
/* ------------------------------------------------------------------ */

static void plugin_exit(qemu_plugin_id_t id, void *p) {
  g_autoptr(GString) out = g_string_new(NULL);

  for (int i = 0; i < qemu_plugin_num_vcpus(); i++) {
    g_string_append_printf(out, "cpu %d insns: %" PRIu64 "\n", i,
                           qemu_plugin_u64_get(insn_count, i));
  }
  g_string_append_printf(out, "total insns: %" PRIu64 "\n",
                         qemu_plugin_u64_sum(insn_count));
  qemu_plugin_outs(out->str);

  qemu_plugin_scoreboard_free(insn_count.score);
  qemu_plugin_scoreboard_free(section_states);
}

/* ------------------------------------------------------------------ */
/* Plugin entry point                                                   */
/* ------------------------------------------------------------------ */

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info, int argc,
                                           char **argv) {
  if (argc > 0) {
    fprintf(stderr, "inscount-profile: no options supported\n");
    return -1;
  }

  insn_count =
      qemu_plugin_scoreboard_u64(qemu_plugin_scoreboard_new(sizeof(uint64_t)));
  section_states = qemu_plugin_scoreboard_new(sizeof(SectionState));

  qemu_plugin_outs("plugin start\n");
  qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);
  qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
  return 0;
}