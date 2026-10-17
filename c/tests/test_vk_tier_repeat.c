/* Isolated policy proof, without Vulkan or a GPU. */
#include "../vk_tier_repeat.h"
#include <stdio.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);return 2;}}while(0)
int main(void) {
    VktRepeatState s={0},other={0};
    CHECK(!vkt_repeat_ready(&s));
    vkt_repeat_record(&s,1);
    CHECK(!vkt_repeat_ready(&s));
    vkt_repeat_record(&s,1);vkt_repeat_record(&s,1);
    CHECK(s.distinct_forwards==1);
    vkt_repeat_record(&other,1);
    CHECK(!vkt_repeat_ready(&other));
    vkt_repeat_record(&s,2);
    CHECK(vkt_repeat_ready(&s));
    CHECK(!vkt_repeat_ready(&other));
    vkt_repeat_record(&s,2);vkt_repeat_record(&s,3);
    CHECK(s.distinct_forwards==2);
    /* Serial 0 always refuses promotion and cannot count as a forward. */
    VktRepeatState z={0};vkt_repeat_record(&z,0);
    CHECK(!vkt_repeat_ready(&z));
    CHECK(!vkt_repeat_ready(NULL));
    puts("PASS causal repeat-first promotion gating with distinct forward epochs");
    return 0;
}
