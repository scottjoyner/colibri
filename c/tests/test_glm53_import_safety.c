#include "../glm53_import_safety.h"
#include <stdio.h>
#include <math.h>
#define ASSERT(b) do {if(!(b)){fprintf(stderr,"NO_GO import guard %d\n",__LINE__);return 2;}}while(0)
int main(void){
    ASSERT(glm53_import_decision(15.84,"1","1",NULL,NULL)==G53_IMPORT_DISABLE);
    ASSERT(glm53_import_decision(15.84,"1","1","1",NULL)==G53_IMPORT_DENY);
    ASSERT(glm53_import_decision(15.84,"1","1","1","1")==G53_IMPORT_OVERRIDE);
    ASSERT(glm53_import_decision(15.84,"1","1","0",NULL)==G53_IMPORT_KEEP);
    ASSERT(glm53_import_decision(15.84,"1","1","true",NULL)==G53_IMPORT_DENY);
    ASSERT(glm53_import_decision(15.84,"1","0","1",NULL)==G53_IMPORT_KEEP);
    ASSERT(glm53_import_decision(15.84,"0","1","1",NULL)==G53_IMPORT_KEEP);
    ASSERT(glm53_import_decision(95.0,"1","1","1",NULL)==G53_IMPORT_KEEP);
    ASSERT(glm53_import_decision(0.0,"1","1",NULL,NULL)==G53_IMPORT_DISABLE);
    ASSERT(glm53_import_decision(NAN,"1","1","1",NULL)==G53_IMPORT_DENY);
    ASSERT(glm53_import_decision(17.179869184,"1","1","1",NULL)==G53_IMPORT_DENY);
    puts("PASS GLM53 small-host staged upload import deny/opt-in");
    return 0;
}
