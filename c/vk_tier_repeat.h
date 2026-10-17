/* Optional repeat-first Vulkan expert promotion: causal and model-independent.
 * A label becomes promotable only after being routed in TWO DISTINCT forwards.
 * No future lookahead, training trace or persistent external file is used.
 * This helper has no allocations and is called only by the tier engine thread.
 */
#ifndef COLIBRI_VK_TIER_REPEAT_H
#define COLIBRI_VK_TIER_REPEAT_H
#include <stdint.h>
typedef struct {
    uint64_t last_forward;
    uint8_t distinct_forwards; /* saturates at two */
} VktRepeatState;
static inline void vkt_repeat_record(VktRepeatState *state, uint64_t serial) {
    if (!state || !serial || state->last_forward==serial)return;
    state->last_forward=serial;
    if(state->distinct_forwards<2)state->distinct_forwards++;
}
static inline int vkt_repeat_ready(const VktRepeatState *state) {
    return state && state->distinct_forwards>=2;
}
#endif
