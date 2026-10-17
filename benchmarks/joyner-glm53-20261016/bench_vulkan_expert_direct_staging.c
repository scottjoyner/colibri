/* Double-buffered 4 MiB GLM expert upload: read next O_DIRECT while Vulkan copies prior.
 * Isolated microbenchmark only, no model serving and no production integration. */
#define main sequential_reference_main
#include "bench_vulkan_expert_transfer.c"
#undef main
#include <pthread.h>
typedef struct {
    GPU *gpu;
    Buffer slots[2], dst, verify_back;
    int verify_every, verified;
    void *scratch[2];
    pthread_mutex_t mutex;
    pthread_cond_t cv;
    int state[2]; /* 0 reusable, 1 ready */
    int n;
    double *gpu_ms;
} Transfer;
static void *uploader(void *v) {
    Transfer *s=v;
    for(int i=0;i<s->n;i++){
        int j=i&1;
        pthread_mutex_lock(&s->mutex);
        while(s->state[j]!=1)pthread_cond_wait(&s->cv,&s->mutex);
        pthread_mutex_unlock(&s->mutex);
        double t=now();
        submit_copy(s->gpu,s->slots[j].b,s->dst.b);
        s->gpu_ms[i]=(now()-t)*1000.;
        if (s->verify_every) {
            submit_copy(s->gpu,s->dst.b,s->verify_back.b);
            if (memcmp(s->verify_back.ptr,s->slots[j].ptr,CHUNK)) {
                fprintf(stderr,"NO_GO GPU corruption on expert transfer %d\n",i);
                _exit(30);
            }
            s->verified++;
        }
        pthread_mutex_lock(&s->mutex);
        s->state[j]=0;
        pthread_cond_broadcast(&s->cv);
        pthread_mutex_unlock(&s->mutex);
    }
    return NULL;
}
int main(int argc,char **argv) {
    if(argc!=4 && argc!=5){fprintf(stderr,"usage: %s <verified-model-shard-file> <aligned-offsets-file> <count:32..512> [--verify-every]\n",argv[0]);return 2;}
    int verify_every=argc==5 && !strcmp(argv[4],"--verify-every");
    if(argc==5 && !verify_every)return 2;
    int n=atoi(argv[3]);if(n<32 || n>MAX_SAMPLES)return 2;
    struct stat st;if(stat(argv[1],&st) || !S_ISREG(st.st_mode) || st.st_size<CHUNK)return 2;
    FILE *list=fopen(argv[2],"r");if(!list)return 2;
    uint64_t off[MAX_SAMPLES];
    for(int i=0;i<n;i++){
        if(fscanf(list,"%" SCNu64,&off[i])!=1 || off[i]%4096 ||
            off[i]>(uint64_t)st.st_size-CHUNK){fputs("NO_GO invalid offset\n",stderr);return 2;}
    }
    fclose(list);
    int fd=open(argv[1],O_RDONLY|O_DIRECT|O_CLOEXEC);
    if(fd<0){perror("open O_DIRECT");return 2;}
    GPU g=gpu_init();
    Transfer x={.gpu=&g,.n=n,.verify_every=verify_every};
    x.dst=allocbuf(&g,CHUNK,VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer back=allocbuf(&g,CHUNK,VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if(verify_every)x.verify_back=allocbuf(&g,CHUNK,VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    for(int i=0;i<2;i++){
        x.slots[i]=allocbuf(&g,CHUNK,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if(((uintptr_t)x.slots[i].ptr & 4095)!=0){
            fprintf(stderr,"NO_GO GPU mapped staging is not aligned for O_DIRECT\n");return 7;
        }
    }
    double *io=calloc(n,sizeof(double)),*stage=calloc(n,sizeof(double));
    x.gpu_ms=calloc(n,sizeof(double));
    if(!io||!stage||!x.gpu_ms)return 3;
    pthread_mutex_init(&x.mutex,NULL);pthread_cond_init(&x.cv,NULL);
    pthread_t thread;
    double t0=now();
    if(pthread_create(&thread,NULL,uploader,&x)){perror("pthread_create");return 3;}
    for(int i=0;i<n;i++){
        int j=i&1;
        pthread_mutex_lock(&x.mutex);
        while(x.state[j]!=0)pthread_cond_wait(&x.cv,&x.mutex);
        /* Claim this free slot; only the producer owns it while updating. */
        x.state[j]=2;
        pthread_mutex_unlock(&x.mutex);
        double t=now();
        ssize_t nr=pread(fd,x.slots[j].ptr,CHUNK,(off_t)off[i]);
        io[i]=(now()-t)*1000.;
        if(nr!=CHUNK){fprintf(stderr,"NO_GO O_DIRECT-to-mapped-staging read=%zd errno=%d (%s)\n",nr,errno,strerror(errno));return 4;}
        stage[i]=0.0;
        pthread_mutex_lock(&x.mutex);
        x.state[j]=1;
        pthread_cond_broadcast(&x.cv);
        pthread_mutex_unlock(&x.mutex);
    }
    pthread_join(thread,NULL);
    double wall=now()-t0;
    if(verify_every && x.verified!=n){fprintf(stderr,"NO_GO verification incomplete %d/%d\n",x.verified,n);return 6;}
    /* After uploader's last fence, the last uploaded tensor must round-trip. */
    submit_copy(&g,x.dst.b,back.b);
    if(memcmp(back.ptr,x.slots[(n-1)&1].ptr,CHUNK)){
        fputs("NO_GO double-buffer GPU byte-for-byte roundtrip mismatch\n",stderr);return 5;
    }
    double i50=quant(io,n,.5),i95=quant(io,n,.95),i99=quant(io,n,.99);
    double s50=quant(stage,n,.5),s95=quant(stage,n,.95),s99=quant(stage,n,.99);
    double g50=quant(x.gpu_ms,n,.5),g95=quant(x.gpu_ms,n,.95),g99=quant(x.gpu_ms,n,.99);
    printf("{\"mode\":\"direct_staging_double_buffer\",\"direct_into_Vulkan\":true,\"verified_uploads\":%d,\"read_source\":\"actual verified GLM expert tensors\","
           "\"samples\":%d,\"bytes_each\":%u,"
           "\"direct_read_ms\":{\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f},"
           "\"cpu_stage_ms\":{\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f},"
           "\"vulkan_copy_ms\":{\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f},"
           "\"pipeline_GBps\":%.3f,\"physical_io_GB\":%.3f,"
           "\"readback_bytes_match\":true,\"Vulkan_gpu\":\"AMD_1002_7551\","
           "\"transfer_parallelism\":2}\n",
           x.verified,n,CHUNK,i50,i95,i99,s50,s95,s99,g50,g95,g99,
           (double)n*CHUNK/wall/1e9,(double)n*CHUNK/1e9);
    pthread_cond_destroy(&x.cv);pthread_mutex_destroy(&x.mutex);
    for(int i=0;i<2;i++){free(x.scratch[i]);release(&g,&x.slots[i]);}
    if(verify_every)release(&g,&x.verify_back);
    release(&g,&back);release(&g,&x.dst);
    vkDestroyFence(g.dev,g.fence,NULL);vkDestroyCommandPool(g.dev,g.pool,NULL);
    vkDestroyDevice(g.dev,NULL);vkDestroyInstance(g.instance,NULL);
    free(io);free(stage);free(x.gpu_ms);close(fd);
    return 0;
}