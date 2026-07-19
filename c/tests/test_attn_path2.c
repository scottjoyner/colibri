/* Standalone correctness test for the Vulkan PATH-2 MLA attention
 * (coli_cuda_attention_project_batch: reconstruct kvb, attend, o_proj).
 * Compares GPU output against an independent CPU reference.
 *   build: gcc -O2 -DCOLI_CUDA test_attn_path2.c backend_vulkan.o -o test_attn_path2 -lvulkan -lpthread -lm -fopenmp
 *   run:   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json ./test_attn_path2
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "backend_cuda.h"

/* weight packing: int4, 2 nibbles/byte, signed (nibble-8). W[o,i] at byte (o*rb + i/2),
 * low nibble = even i, high nibble = odd i.  rb = (I+1)/2. */
static void pack_int4(uint8_t* dst, int O, int I, const float* W, const float* scales){
    int rb=(I+1)/2;
    for(int o=0;o<O;o++){
        for(int i=0;i<I;i+=2){
            int lo=(int)lrintf(W[(size_t)o*I+i])+8;
            int hi=(i+1<I)?(int)lrintf(W[(size_t)o*I+i+1])+8:8;
            if(lo<0)lo=0; if(lo>15)lo=15;
            if(hi<0)hi=0; if(hi>15)hi=15;
            dst[(size_t)o*rb+(i>>1)]=(uint8_t)(lo | (hi<<4));
        }
    }
    (void)scales;
}
static int nib(uint8_t b,int k){ int n=(k&1)?(b>>4)&0xF:b&0xF; return n-8; }

/* CPU reference: y = o_proj( attend( reconstruct(kv_b, Lc), q, rope ) ) */
static void cpu_ref(const uint8_t* Wq, const float* sq, const uint8_t* Wo, const float* so,
                    const float* q, const float* l, const float* rope,
                    int S,int H,int Q,int R,int V,int K,int T,float scale,
                    float* out){
    int kvb_dim=H*(Q+V); int rb=(K+1)/2;
    float* kvb=malloc((size_t)T*kvb_dim*sizeof(float));
    for(int t=0;t<T;t++) for(int rr=0;rr<kvb_dim;rr++){
        float a=0; for(int d=0;d<K;d++) a+=l[(size_t)t*K+d]*nib(Wq[(size_t)rr*rb+(d>>1)],d)*sq[rr];
        kvb[(size_t)t*kvb_dim+rr]=a;
    }
    float* ctx=malloc((size_t)S*H*V*sizeof(float));
    for(int s=0;s<S;s++){ int nt=T-S+s+1;
        for(int h=0;h<H;h++){ int rbase=h*(Q+V); int qoff=(s*H+h)*(Q+R);
            float* scv=malloc(nt*sizeof(float)); float mx=-1e30f;
            for(int t=0;t<nt;t++){ float a=0;
                for(int k=0;k<Q;k++) a+=q[qoff+k]*kvb[(size_t)t*kvb_dim+rbase+k];
                for(int d=0;d<R;d++) a+=q[qoff+Q+d]*rope[(size_t)t*R+d];
                scv[t]=a*scale; if(scv[t]>mx)mx=scv[t]; }
            float sum=0; for(int t=0;t<nt;t++){scv[t]=expf(scv[t]-mx);sum+=scv[t];}
            float inv=1.0f/(sum>0?sum:1.0f);
            for(int v=0;v<V;v++){ float a=0;
                for(int t=0;t<nt;t++) a+=scv[t]*inv*kvb[(size_t)t*kvb_dim+rbase+Q+v];
                ctx[((size_t)s*H+h)*V+v]=a; }
            free(scv);
        }
    }
    /* o_proj: out[S,D] = ctx[S,H*V] @ Wo[H*V, D] (D=H*V here) */
    int D=H*V; int orb=(D+1)/2;
    for(int s=0;s<S;s++) for(int d=0;d<D;d++){
        float a=0; for(int hv=0;hv<D;hv++) a+=ctx[((size_t)s*D)+hv]*nib(Wo[(size_t)d*orb+(hv>>1)],hv)*so[d];
        out[(size_t)s*D+d]=a;
    }
    free(kvb); free(ctx);
}

int main(void){
    int H=64,Q=192,R=64,V=256,K=512,S=8,T=48;
    int kvb_dim=H*(Q+V); int D=H*V;
    int dev=0;
    if(!coli_cuda_init(&dev,1)){ fprintf(stderr,"init failed\n"); return 1; }

    /* kv_b weights: O=H*(Q+V), I=K */
    int O=kvb_dim;
    float* Wq=malloc(O*K*sizeof(float)); float* sq=malloc(O*sizeof(float));
    for(int i=0;i<O*K;i++) Wq[i]=((float)rand()/RAND_MAX*2-1)*4;
    for(int i=0;i<O;i++) sq[i]=0.5f+((float)rand()/RAND_MAX);
    uint8_t* Wq_p=malloc((size_t)O*((K+1)/2)); pack_int4(Wq_p,O,K,Wq,sq);

    /* o_proj weights: O=D, I=H*V=D */
    float* Wo=malloc(D*D*sizeof(float)); float* so=malloc(D*sizeof(float));
    for(int i=0;i<D*D;i++) Wo[i]=((float)rand()/RAND_MAX*2-1)*4;
    for(int i=0;i<D;i++) so[i]=0.5f+((float)rand()/RAND_MAX);
    uint8_t* Wo_p=malloc((size_t)D*((D+1)/2)); pack_int4(Wo_p,D,D,Wo,so);

    ColiCudaTensor *kv=NULL,*o=NULL;
    if(!coli_cuda_tensor_upload(&kv,Wq_p,sq,2,K,O,dev)){ fprintf(stderr,"kv upload failed\n"); return 1; }
    if(!coli_cuda_tensor_upload(&o,Wo_p,so,2,D,D,dev)){ fprintf(stderr,"o upload failed\n"); return 1; }

    float* q=malloc((size_t)S*H*(Q+R)*sizeof(float));
    float* l=malloc((size_t)T*K*sizeof(float));
    float* rope=malloc((size_t)T*R*sizeof(float));
    float* gpu=malloc((size_t)S*D*sizeof(float));
    float* ref=malloc((size_t)S*D*sizeof(float));
    for(int i=0;i<S*H*(Q+R);i++) q[i]=((float)rand()/RAND_MAX*2-1);
    for(int i=0;i<T*K;i++) l[i]=((float)rand()/RAND_MAX*2-1);
    for(int i=0;i<T*R;i++) rope[i]=((float)rand()/RAND_MAX*2-1);

    float scale=1.0f/sqrtf((float)Q);
    float* gpu2=malloc((size_t)S*D*sizeof(float));
    /* absorb (o=NULL) FIRST, then project_batch (o set) — the engine's decode->prefill pattern */
    if(!coli_cuda_attention_absorb_batch(kv,gpu2,q,l,rope,S,H,Q,R,V,K,T,scale)){
        fprintf(stderr,"absorb_batch failed\n"); return 1;
    }
    if(!coli_cuda_attention_project_batch(kv,o,gpu,q,l,rope,S,H,Q,R,V,K,T,scale)){
        fprintf(stderr,"project_batch failed\n"); return 1;
    }
    double n1=0,n2=0; for(int i=0;i<S*D;i++){ n1+=gpu[i]*gpu[i]; n2+=gpu2[i]*gpu2[i]; }
    printf("project(o) norm=%.4f absorb(o=NULL) norm=%.4f\n", sqrt(n1), sqrt(n2));
    cpu_ref(Wq_p,sq,Wo_p,so,q,l,rope,S,H,Q,R,V,K,T,scale,ref);
    float md=0; int intm=0; float reld=0, rel10=0;
    for(int i=0;i<S*D;i++){ float d=fabsf(gpu[i]-ref[i]); if(d>md){md=d;intm=i;}
        if(fabsf(ref[i])>1.0f){ float rd=d/fabsf(ref[i]); if(rd>reld)reld=rd; }
        if(fabsf(ref[i])>10.0f){ float rd=d/fabsf(ref[i]); if(rd>rel10)rel10=rd; } }
    printf("PATH-2 GPU vs CPU ref: S=%d H=%d Q=%d R=%d V=%d K=%d T=%d scale=%.5f\n",S,H,Q,R,V,K,T,scale);
    printf("maxabs=%.6f maxrel(>10)=%.6e at %d  gpu=%.6f ref=%.6f\n", md, rel10, intm, gpu[intm], ref[intm]);
    int ok = (md < 1.0f) && (rel10 < 5e-3);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok?0:2;
}
