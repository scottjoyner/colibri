/* Real preadv scatter correctness, partial-EOF failure, bounded iovec guard. */
#define _GNU_SOURCE
#include "../glm53_expert_read_plan.h"
#include "../glm53_expert_preadv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#define REQUIRE(c) do {if(!(c)){fprintf(stderr,"NO_GO %d\n",__LINE__);return 2;}}while(0)
int main(void) {
    char name[]="/tmp/glm53-scatter-XXXXXX";
    int fd=mkstemp(name);REQUIRE(fd>=0);REQUIRE(!unlink(name));
    unsigned char src[512],dst[512];
    for(int i=0;i<512;i++)src[i]=(unsigned char)(i*13+19);
    REQUIRE(write(fd,src,512)==512);
    int refs[6]={fd,fd,fd,fd,fd,fd};
    int64_t off[6]={256,288,352,384,128,192},len[6]={32,64,32,64,64,64};
    /* The logical parts are scattered, and the six source tensors make one extent.
     * No gaps are read into the destination buffers. */
    G53IOplan p;REQUIRE(g53_io_plan(refs,off,len,320,&p));
    REQUIRE(p.nruns==1);
    unsigned char out[6][64]={0};
    for(int k=0;k<p.nruns;k++){
        G53IOrun *r=&p.run[k];
        int order[6]={0,1,2,3,4,5};
        for(int a=1;a<6;a++){int j=a,x=order[a];while(j>0&&p.piece_dest[order[j-1]]>p.piece_dest[x]){order[j]=order[j-1];j--;}order[j]=x;}
        struct iovec vec[6];int n=0;size_t bytes=0;
        for(int a=0;a<6;a++){
            int t=order[a];
            if(p.piece_dest[t]>=r->dest&&p.piece_dest[t]<r->dest+(size_t)r->len){
                vec[n++]=(struct iovec){out[t],(size_t)len[t]};bytes+=(size_t)len[t];
            }
        }
        unsigned int calls=0;
        REQUIRE(bytes==(size_t)r->len);
        REQUIRE(g53_preadv_full(fd,vec,n,r->off,bytes,&calls));
        REQUIRE(calls==1);
    }
    for(int i=0;i<6;i++)REQUIRE(!memcmp(out[i],src+off[i],(size_t)len[i]));
    struct iovec bad={(void *)dst,64};
    REQUIRE(!g53_preadv_full(fd,&bad,1,504,64,NULL));
    REQUIRE(!g53_preadv_full(fd,&bad,1,-1,64,NULL));
    REQUIRE(!g53_preadv_full(fd,&bad,1,0,65,NULL));
    REQUIRE(!close(fd));
    puts("PASS native Linux preadv scatter, EOF and bounds");
    return 0;
}
