/* bench_attn.c -- measure Vulkan PATH-2 attention throughput (ms/call) for
 * realistic GLM-5.2 shapes at a deep context (T=2048). Covers the R (reconstruct)
 * + S (attend) shaders. Build:
 *   gcc -O2 -DCOLI_CUDA bench_attn.c backend_vulkan.o -o bench_attn -lvulkan -lpthread -lm -fopenmp
 * Run: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json ./bench_attn
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "backend_cuda.h"
static double now_s(void){ struct timeval tv; gettimeofday(&tv,NULL); return tv.tv_sec+tv.tv_usec*1e-6; }

int main(void){
    int H=64,Q=192,R=64,V=256,K=512;
    int dev=0; if(!coli_cuda_init(&dev,1)){ fprintf(stderr,"init failed\n"); return 1; }
    int shapes[][2]={{1,256},{1,1024},{1,2048},{7,2048},{28,2048}};
    int n=sizeof(shapes)/sizeof(shapes[0]);
    for(int si=0;si<n;si++){
        int S=shapes[si][0], T=shapes[si][1];
        fprintf(stderr,"[bench] S=%d T=%d start\n",S,T);
        int kvb_dim=H*(Q+V), O=kvb_dim;
        int rb=(K+1)/2;
        uint8_t* Wq=malloc((size_t)O*rb); float* sq=malloc(O*sizeof(float));
        float* q=malloc((size_t)S*H*(Q+R)*sizeof(float));
        float* l=malloc((size_t)T*K*sizeof(float));
        float* rope=malloc((size_t)T*R*sizeof(float));
        float* out=malloc((size_t)S*H*V*sizeof(float));
        for(size_t i=0;i<(size_t)O*rb;i++) Wq[i]=(uint8_t)(rand()&0xFF);
        for(int i=0;i<O;i++) sq[i]=0.5f;
        for(size_t i=0;i<(size_t)S*H*(Q+R);i++) q[i]=(float)(rand()&0xFF)/255.0f-0.5f;
        for(size_t i=0;i<(size_t)T*K;i++) l[i]=(float)(rand()&0xFF)/255.0f-0.5f;
        for(size_t i=0;i<(size_t)T*R;i++) rope[i]=(float)(rand()&0xFF)/255.0f-0.5f;
        ColiCudaTensor* kv=NULL;
        if(!coli_cuda_tensor_upload(&kv,Wq,sq,2,K,O,dev)){ fprintf(stderr,"upload failed\n"); return 1; }
        float scale=1.0f/sqrtf((float)Q);
        /* warmup */
        fprintf(stderr,"[bench]   upload+warmup...\n");
        coli_cuda_attention_absorb_batch(kv,out,q,l,rope,S,H,Q,R,V,K,T,scale);
        fprintf(stderr,"[bench]   warmup done\n");
        int N=3; double t0=now_s();
        for(int it=0;it<N;it++){ double t1=now_s();
            coli_cuda_attention_absorb_batch(kv,out,q,l,rope,S,H,Q,R,V,K,T,scale);
            fprintf(stderr,"[bench]   iter %d: %.1f ms\n", it, (now_s()-t1)*1000.0); }
        double dt=now_s()-t0;
        /* MACs: reconstruction O*K*T + attend S*H*V*T*(Q+R) */
        double macs=(double)O*K*T + (double)S*H*V*T*(Q+R);
        printf("S=%-3d T=%-4d : %.2f ms/call  %.1f GMAC/s\n", S,T, dt/N*1000.0, macs*N/dt/1e9);
        free(Wq);free(sq);free(q);free(l);free(rope);free(out);
    }
    return 0;
}
