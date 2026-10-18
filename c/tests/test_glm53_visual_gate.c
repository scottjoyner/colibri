#include "../glm53_visual_gate.h"
#include <stdio.h>
#define ASSERT(x) do{if(!(x)){fprintf(stderr,"FAIL GLM visual gate line %d\n",__LINE__);return 2;}}while(0)
int main(void){
    const int visual[]={268,154830,154831,154855,154832,154833};
    int text[]={11,21,13},image[]={11,268,13},video[]={154855};
    ASSERT(!glm53_visual_ids_present(visual,text,3));
    ASSERT(glm53_visual_ids_present(visual,image,3));
    ASSERT(glm53_visual_ids_present(visual,video,1));
    for(int i=0;i<6;i++){
        int one[]={visual[i]};
        ASSERT(glm53_visual_ids_present(visual,one,1));
    }
    ASSERT(glm53_visual_ids_present(visual,NULL,3)); /* deny malformed */
    ASSERT(glm53_visual_ids_present(NULL,text,3));
    ASSERT(glm53_visual_ids_present(visual,text,-1));
    const int missing[]={-1,-1,-1,-1,-1,-1};
    ASSERT(!glm53_visual_ids_present(missing,text,3));
    puts("PASS GLM text-only visual placeholders, all six markers fail closed");
    return 0;
}
