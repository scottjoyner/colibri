/* Standalone byte-for-byte source fixture for the native six-tensor planner.
 * gcc -O2 -std=gnu11 -Wall -Wextra -Werror c/tests/test_glm53_expert_read_plan.c -o /tmp/plan && /tmp/plan */
#define _GNU_SOURCE
#include "../glm53_expert_read_plan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#define CHECK(c) do {if(!(c)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#c);exit(1);}}while(0)

static const int seq[6]={4,5,0,1,2,3};
static const int64_t sizes[6]={256,32,256,32,256,32};
static unsigned char expected[6][256];

static void scenario(int mode) {
    char name1[]="/tmp/g53-read-plan-A-XXXXXX";
    char name2[]="/tmp/g53-read-plan-B-XXXXXX";
    int f[2]={mkstemp(name1),mkstemp(name2)};
    CHECK(f[0]>=0&&f[1]>=0);
    CHECK(!unlink(name1)&&!unlink(name2));
    int fd[6];int64_t off[6];
    int64_t place[2]={97,87};
    for(int k=0;k<6;k++){
        int p=seq[k];
        int source=mode==4 ? (k>=3 ? 1:0) : 0;
        if (mode==2 && k==3)place[source]+=19;
        if (mode==4 && (k==2||k==5))place[source]+=19;
        if (mode==6 && k)place[source]+=19;
        fd[p]=f[source];
        off[p]=place[source];
        for(int j=0;j<sizes[p];j++)expected[p][j]=(unsigned char)(3+p*31+j*7);
        CHECK(pwrite(f[source],expected[p],(size_t)sizes[p],off[p])==sizes[p]);
        place[source]+=sizes[p];
    }
    G53IOplan plan;
    size_t sum=0;for(int p=0;p<6;p++)sum+=(size_t)sizes[p];
    CHECK(g53_io_plan(fd,off,sizes,sum,&plan));
    CHECK(plan.nruns==mode);
    CHECK(plan.total_bytes==sum);
    unsigned char *buf=malloc(sum);
    CHECK(buf!=NULL);
    memset(buf,0x55,sum);
    for(int i=0;i<plan.nruns;i++){
        const G53IOrun *r=&plan.run[i];
        CHECK(r->len>0&&r->dest+ (size_t)r->len<=sum);
        CHECK(pread(r->fd,buf+r->dest,(size_t)r->len,r->off)==r->len);
    }
    for(int p=0;p<6;p++){
        CHECK(plan.piece_dest[p]+(size_t)sizes[p]<=sum);
        CHECK(!memcmp(buf+plan.piece_dest[p],expected[p],(size_t)sizes[p]));
    }
    free(buf);
    close(f[0]);close(f[1]);
}

static void negative_cases(void) {
    int fd[6]={10,10,10,10,10,10};
    int64_t off[6]={0,8,16,24,32,40};
    int64_t len[6]={8,8,8,8,8,8};
    G53IOplan p;
    CHECK(g53_io_plan(fd,off,len,48,&p));
    CHECK(p.nruns==1);
    CHECK(!g53_io_plan(fd,off,len,47,&p));
    CHECK(!g53_io_plan(fd,off,len,49,&p));
    off[2]=9;
    CHECK(!g53_io_plan(fd,off,len,48,&p));
    off[2]=16;off[0]=-1;
    CHECK(!g53_io_plan(fd,off,len,48,&p));
    off[0]=INT64_MAX-7;
    CHECK(!g53_io_plan(fd,off,len,48,&p));
    off[0]=0; fd[3]=-1;
    CHECK(!g53_io_plan(fd,off,len,48,&p));
    fd[3]=10;len[1]=0;
    CHECK(!g53_io_plan(fd,off,len,48,&p));
    CHECK(!g53_io_plan(fd,off,len,48,NULL));
}

int main(void) {
    scenario(1);scenario(2);scenario(4);scenario(6);
    negative_cases();
    puts("PASS native GLM53 grouped 1/2/4/6-run payload parity, reject overlaps/overflow/capacity");
    return 0;
}
