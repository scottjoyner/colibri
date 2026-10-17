/* Exact six-tensor buffered pread A/B/C/C/B/A on a private frozen GLM-5.3 holdout.
 * Input is a verified text manifest, never model weights. Production unchanged. */
#define _GNU_SOURCE
#include "../glm53_expert_read_plan.h"
#include "../glm53_expert_preadv.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define CAP 33554432u
#define MAX_EXPERTS 64
#define NEED(x) do {if(!(x)){fprintf(stderr,"NO_GO line %d: %s\n",__LINE__,#x);exit(2);}}while(0)
typedef struct {int fd[6];int64_t off[6],len[6];size_t bytes;} Source;
static Source experts[MAX_EXPERTS];
static uint64_t expected[MAX_EXPERTS][6];
static int fds[62], count;
static unsigned char *scratch;
static double stamp(void) {struct timespec t;NEED(!clock_gettime(CLOCK_MONOTONIC_RAW,&t));return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static int cmp(const void *a,const void *b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double pct(const double *xs,int n,double q){double v[MAX_EXPERTS];memcpy(v,xs,(size_t)n*sizeof(double));qsort(v,n,sizeof(double),cmp);int p=(int)ceil(n*q)-1;if(p<0)p=0;if(p>=n)p=n-1;return v[p];}
static uint64_t digest(const void *buf,size_t size){uint64_t h=UINT64_C(14695981039346656037);const unsigned char *p=buf;for(size_t k=0;k<size;k++){h^=p[k];h*=UINT64_C(1099511628211);}return h;}
static void all_read(int fd,void *output,size_t n,int64_t offset){
    unsigned char *dst=output;
    for(size_t at=0;at<n;) {
        ssize_t got=pread(fd,dst+at,n-at,offset+(int64_t)at);
        if(got<0&&errno==EINTR)continue;
        NEED(got>0);at+=(size_t)got;
    }
}
static void manifest(const char *path){
    FILE *f=fopen(path,"r");NEED(f);
    NEED(fscanf(f,"%d",&count)==1 && count>=16 && count<=MAX_EXPERTS);
    for(int i=0;i<62;i++)fds[i]=-1;
    size_t reference=0;
    for(int i=0;i<count;i++){
        size_t total=0;
        for(int p=0;p<6;p++){
            int shard;long long off,len;
            NEED(fscanf(f,"%d %lld %lld",&shard,&off,&len)==3);
            NEED(shard>=1&&shard<=62&&off>=0&&len>0&&len<=CAP);
            NEED(total<=CAP-(size_t)len);
            if(fds[shard-1]<0){
                char filename[256];
                snprintf(filename,sizeof(filename),
                 "/srv/joyner-gen5/models/glm53_colibri_i4/model-%05d-of-00062.safetensors",shard);
                fds[shard-1]=open(filename,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
                NEED(fds[shard-1]>=0);
            }
            struct stat st;NEED(!fstat(fds[shard-1],&st)&&S_ISREG(st.st_mode));
            NEED(off<=st.st_size&&len<=st.st_size-off);
            experts[i].fd[p]=fds[shard-1];
            experts[i].off[p]=off;experts[i].len[p]=len;
            total+=(size_t)len;
        }
        experts[i].bytes=total;
        if(!i)reference=total;
        NEED(total==reference);
    }
    NEED(!fclose(f));
}
static void replay(int mode,int calibrate){
    double elapsed[MAX_EXPERTS]={0};
    uint64_t calls=0;
    for(int i=0;i<count;i++){
        const Source *e=&experts[i];
        G53IOplan p={0};
        if(mode)NEED(g53_io_plan(e->fd,e->off,e->len,e->bytes,&p));
        double t=stamp();
        if(mode==2){
            size_t logical[6];size_t at=0;
            for(int j=0;j<6;j++){logical[j]=at;at+=(size_t)e->len[j];}
            int order[6]={0,1,2,3,4,5};
            for(int k=1;k<6;k++){
                int x=order[k],j=k;
                while(j>0&&p.piece_dest[order[j-1]]>p.piece_dest[x]){
                    order[j]=order[j-1];j--;
                }
                order[j]=x;
            }
            for(int j=0;j<p.nruns;j++){
                G53IOrun *r=&p.run[j];
                struct iovec iov[6];int n=0;size_t packed=0;
                for(int k=0;k<6;k++){
                    int piece=order[k];size_t pos=p.piece_dest[piece];
                    if(pos>=r->dest&&pos<r->dest+(size_t)r->len){
                        NEED(pos==r->dest+packed);
                        iov[n++]=(struct iovec){scratch+logical[piece],(size_t)e->len[piece]};
                        packed+=(size_t)e->len[piece];
                    }
                }
                NEED(packed==(size_t)r->len);
                unsigned int nreads=0;
                NEED(g53_preadv_full(r->fd,iov,n,r->off,(size_t)r->len,&nreads));
                calls+=nreads;
            }
        }else if(mode==1){
            for(int j=0;j<p.nruns;j++){
                G53IOrun *r=&p.run[j];
                all_read(r->fd,scratch+r->dest,(size_t)r->len,r->off);
            }
            calls+=(uint64_t)p.nruns;
        }else{
            size_t off=0;
            for(int j=0;j<6;j++){
                all_read(e->fd[j],scratch+off,(size_t)e->len[j],e->off[j]);
                off+=(size_t)e->len[j];
            }
            calls+=6;
        }
        elapsed[i]=stamp()-t;
        size_t off=0;
        for(int j=0;j<6;j++){
            size_t start=mode==1?p.piece_dest[j]:off;
            uint64_t value=digest(scratch+start,(size_t)e->len[j]);
            if(calibrate)expected[i][j]=value;
            else NEED(expected[i][j]==value);
            off+=(size_t)e->len[j];
        }
    }
    printf("{\"mode\":\"%s\",\"experts\":%d,\"read_calls\":%llu,"
           "\"p50_ms\":%.3f,\"p95_ms\":%.3f,\"p99_ms\":%.3f,"
           "\"all_six_tensor_hashes_match\":true}\n",
           mode==2?"vector":(mode==1?"grouped":"six"),count,(unsigned long long)calls,
           pct(elapsed,count,.5),pct(elapsed,count,.95),pct(elapsed,count,.99));
    fflush(stdout);
}
int main(int argc,char **argv){
    if(argc!=2){fputs("usage: bench_glm53_native_grouped_shadow MANIFEST\n",stderr);return 2;}
    manifest(argv[1]);scratch=malloc(CAP);NEED(scratch);
    replay(0,1);replay(1,0);replay(2,0);replay(2,0);replay(1,0);replay(0,0);
    for(int i=0;i<62;i++)if(fds[i]>=0)NEED(!close(fds[i]));
    free(scratch);puts("PASS native buffered read 6-pass exact tensor checks");
    return 0;
}
