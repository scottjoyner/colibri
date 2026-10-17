/* Pure causal per-forward fair admission quota across MoE layers.
 * Cap is the existing global per-token promotion budget; rotating a leftover
 * share prevents low-index layers monopolizing the queue every decode step.
 * No knowledge of future expert labels, cross-session traces or writes.
 */
#ifndef COLI_VK_TIER_FAIR_QUOTA_H
#define COLI_VK_TIER_FAIR_QUOTA_H
#include <stdint.h>
static inline int vkt_fair_quota(int global_cap,int first_layer,
                                 int active_layers,int layer,
                                 uint64_t epoch) {
    if(global_cap<=0||first_layer<0||active_layers<=0||
       layer<first_layer||layer-first_layer>=active_layers||!epoch)
        return 0;
    int each=global_cap/active_layers;
    int extras=global_cap%active_layers;
    if(!extras)return each;
    /* Bounded multiplication: reduce the epoch before multiplying.
     * extras < active_layers and active_layers are bounded by the engine
     * geometry, so the product stays inside 32-bit integer arithmetic
     * for normal model configurations. Use uint64_t for full safety. */
    int rotate=(int)(((epoch-1)%(uint64_t)active_layers *
                       (uint64_t)extras)%(uint64_t)active_layers);
    int pos=layer-first_layer;
    int order=(pos+active_layers-rotate)%active_layers;
    return each+(order<extras);
}
#endif
