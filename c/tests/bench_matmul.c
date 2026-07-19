/* bench_matmul.c -- measure Vulkan int4 matmul throughput (GMAC/s) for the
 * shapes that dominate the attention O_proj (I=O=16384, S=7) and decode
 * (S=1). Build:
 *   gcc -O2 -DCOLI_CUDA bench_matmul.c backend_vulkan.o -o bench_matmul -lvulkan -lpthread -lm -fopenmp
 * Run: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json ./bench_matmul
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "backend_cuda.h"

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(void){
    int dev=0;
    if(!coli_cuda_init(&dev,1)){ fprintf(stderr,"coli_cuda_init failed\n"); return 1; }
    /* shapes: {I, O, S} */
    int shapes[][3] = {
        {16384,16384,7},   /* O_proj prefill */
        {16384,16384,1},   /* O_proj decode */
        {6144,2048,7},     /* expert gate/up prefill */
        {6144,2048,1},     /* expert gate/up decode */
        {2048,6144,7},     /* expert down prefill */
    };
    int nsh=sizeof(shapes)/sizeof(shapes[0]);
    for(int s=0;s<nsh;s++){
        int I=shapes[s][0], O=shapes[s][1], S=shapes[s][2];
        size_t wbytes=(size_t)O*((I+1)/2);
        uint8_t* w=malloc(wbytes); float* sc=malloc((size_t)O*sizeof(float));
        float* x=malloc((size_t)S*I*sizeof(float));
        float* y=malloc((size_t)S*O*sizeof(float));
        for(size_t i=0;i<wbytes;i++) w[i]=(uint8_t)(rand()&0xFF);
        for(int i=0;i<O;i++) sc[i]=0.02f;
        for(int i=0;i<S*I;i++) x[i]=(float)(rand()&0xFF)/255.0f-0.5f;
        ColiCudaTensor* t=NULL;
        /* warmup + upload */
        if(!coli_cuda_matmul(&t,y,x,w,sc,2,S,I,O,0)){ fprintf(stderr,"matmul init failed I=%d O=%d S=%d\n",I,O,S); return 1; }
        int N=20; double t0=now_s();
        for(int it=0;it<N;it++) coli_cuda_matmul(&t,y,x,w,sc,2,S,I,O,0);
        double dt=now_s()-t0;
        double macs=(double)S*(double)I*(double)O*(double)N;
        double gmacs=macs/dt/1e9;
        printf("I=%-6d O=%-6d S=%-3d : %.3f ms/call  %.1f GMAC/s\n", I,O,S, dt/N*1000.0, gmacs);
        free(w);free(sc);free(x);free(y);
    }
    return 0;
}
