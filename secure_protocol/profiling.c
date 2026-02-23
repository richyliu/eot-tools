#include "profiling.h"
#include "ext_support.h"

#ifdef TARGET_ARM
void log_stack_usage(void) {
    uint32_t used = stack_get_usage();
    uint32_t total = stack_get_total();
    ext_io_printf("Stack: %u/%u bytes (%u%%)\n", used, total, (used * 100) / total);
}
#endif
