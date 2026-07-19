/* backend_vulkan.c — colibri Vulkan backend for the Radeon 890M / any Vulkan GPU.
 *
 * Implements the SAME ABI as backend_cuda.h (the coli_cuda_* symbols) but
 * against Vulkan instead of CUDA. glm.c's existing #ifdef COLI_CUDA dispatch
 * calls these names unchanged; we build with -DCOLI_CUDA and link this file
 * instead of backend_cuda.cu, so the engine needs zero changes.
 *
 * Scope (first working version):
 *   - coli_cuda_init / shutdown / mem_info / stats / device_count
 *   - coli_cuda_tensor_upload (alloc + upload int4 weights + per-row scales)
 *   - coli_cuda_matmul        (fmt=2 int4; others fall back to CPU via return 0)
 *   - coli_cuda_expert_mlp    (int4 gate/up/silu/down; covers the MoE experts)
 *   - coli_cuda_expert_group  -> returns 0 (glm.c fans out to per-expert expert_mlp)
 *   - all other coli_cuda_*    -> return 0 / no-op (engine keeps CPU path)
 *
 * The int4 math exactly mirrors glm.c matmul_i4: 2 nibbles/byte (low=first,
 * signed -8..+7), one FP32 scale per output row applied to the whole dot.
 *
 * Shaders are compiled at init via glslangValidator (dev build). For a shippable
 * binary, precompile to SPIR-V and embed the byte arrays (see plan doc).
 *
 * Build (colibri/c):
 *   make glm VULKAN=1
 * Run (force the AMD ICD on machines with multiple GPUs):
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json ./glm ...
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <vulkan/vulkan.h>

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

/* ---- opaque tensor (engine stores the pointer, never dereferences) ---- */
struct ColiCudaTensor {
    int device;
    int fmt;            /* 0=f32 1=int8 2=int4 3=int2 */
    int I, O;
    VkBuffer wbuf;      /* packed weights (q4/q8) */
    VkDeviceMemory wmem;
    VkBuffer sbuf;      /* FP32 scales (O floats) */
    VkDeviceMemory smem;
    size_t wbytes, sbytes;
};
typedef struct ColiCudaTensor ColiCudaTensor;

/* ---- Vulkan singleton ---- */
static VkInstance   g_inst = NULL;
static VkPhysicalDevice g_pd = NULL;
static VkDevice     g_dev = NULL;
static VkQueue      g_queue = NULL;
static uint32_t     g_qf = 0;
static VkCommandPool g_pool = NULL;
static VkFence      g_fence = NULL;
static int          g_enabled = 0;
static int          g_devices[16];
static int          g_ndev = 0;

/* stats */
static uint64_t g_calls=0, g_experts=0, g_rows=0;
static double   g_h2d_ms=0, g_kernel_ms=0, g_d2h_ms=0;

/* ---- shaders (GLSL, compiled to SPIR-V at init via glslangValidator) ---- */
/* Tiled int4 GEMM: y[S,O] = x[S,I] @ W[O,I]^T, W int4 (signed nibbles, row scales).
 * One workgroup computes a TILE_S x TILE_O block of y. For each K-tile it
 * cooperatively loads a dequantized W-tile and an x-tile into shared memory
 * (fast, coalesced global reads), then each thread accumulates from shared mem.
 * This removes the catastrophic strided global reads of the naive kernel
 * (which hit ~1 GMAC/s at O=16384 vs ~60 GMAC/s at O=2048). */
#define TM_TILE_S 8
#define TM_TILE_O 8
#define TM_TILE_K 32
#define STR(x) #x
#define XSTR(x) STR(x)
static const char* S_MATMUL =
"#version 450\n"
"layout(local_size_x=" XSTR(TM_TILE_O) ", local_size_y=" XSTR(TM_TILE_S) ") in;\n"
"layout(std430,binding=0) readonly buffer Wbuf { uint w[]; };\n"
"layout(std430,binding=1) readonly buffer Sbuf { float sc[]; };\n"
"layout(std430,binding=2) readonly buffer Xbuf { float x[]; };\n"
"layout(std430,binding=3) writeonly buffer Ybuf { float y[]; };\n"
"shared float smW[" XSTR(TM_TILE_O) "*" XSTR(TM_TILE_K) "];\n"
"shared float smX[" XSTR(TM_TILE_S) "*" XSTR(TM_TILE_K) "];\n"
"int wgt(uint row,uint k){ uint rb=uint((I_DIM+1)/2); uint bo=row*rb+(k>>1u); uint byte=(w[bo>>2u] >> ((bo&3u)*8u)) & 0xFFu; int nib=(k&1u)!=0?int(byte>>4u):int(byte&0xFu); return nib-8; }\n"
"void main(){\n"
"  uint o0=(gl_WorkGroupID.x*" XSTR(TM_TILE_O) ")+gl_LocalInvocationID.x;\n"
"  uint s0=(gl_WorkGroupID.y*" XSTR(TM_TILE_S) ")+gl_LocalInvocationID.y;\n"
"  uint lx=gl_LocalInvocationID.x, ly=gl_LocalInvocationID.y;\n"
"  uint I=uint(I_DIM), O=uint(O_DIM), S=uint(S_DIM);\n"
"  float acc=0.0f;\n"
"  for(uint k0=0u;k0<I;k0+=" XSTR(TM_TILE_K) "){\n"
"    for(uint e=lx+ly*" XSTR(TM_TILE_O) "; e<" XSTR(TM_TILE_O) "*" XSTR(TM_TILE_K) "; e+=" XSTR(TM_TILE_O) "*" XSTR(TM_TILE_S) "){\n"
"      uint tk=e%" XSTR(TM_TILE_K) "; uint to=e/" XSTR(TM_TILE_K) ";\n"
"      uint gk=k0+tk; uint go=o0-lx+to; uint gs=s0-ly+to;\n"
"      if(go<O && tk<(I-k0)) smW[to*" XSTR(TM_TILE_K) "+tk]=float(wgt(go,gk)); else smW[to*" XSTR(TM_TILE_K) "+tk]=0.0f;\n"
"      if(gs<S && tk<(I-k0)) smX[to*" XSTR(TM_TILE_K) "+tk]=x[gs*I+gk]; else smX[to*" XSTR(TM_TILE_K) "+tk]=0.0f;\n"
"    }\n"
"    barrier();\n"
"    if(o0<O && s0<S){\n"
"      for(uint k=0u;k<" XSTR(TM_TILE_K) ";k++) acc += smX[ly*" XSTR(TM_TILE_K) "+k]*smW[lx*" XSTR(TM_TILE_K) "+k];\n"
"    }\n"
"    barrier();\n"
"  }\n"
"  if(o0<O && s0<S) y[s0*O+o0]=acc*sc[o0];\n"
"}\n";

static const char* S_SILU =
"#version 450\n"
"layout(local_size_x=64) in;\n"
"layout(std430,binding=0) buffer Gbuf { float g[]; };\n"
"layout(std430,binding=1) readonly buffer Ubuf { float u[]; };\n"
"void main(){\n"
"  uint i=gl_GlobalInvocationID.x; if(i>=uint(N_DIM)) return;\n"
"  float v=g[i]; g[i]= v/(1.0f+exp(-v)) * u[i];\n"
"}\n";

/* ---- embed constants via simple string substitution at compile time ---- */
/* We bake I/O/S as specialization-free literal by patching the GLSL string.  */
static char* patch_dims(const char* src, int I, int O, int S){
    /* replace I_DIM/O_DIM/S_DIM tokens */
    size_t cap = strlen(src)+64;
    char* out = malloc(cap);
    const char* p = src; int j=0;
    char ibuf[16], obuf[16], sbuf[16];
    snprintf(ibuf,sizeof ibuf,"%d",I); snprintf(obuf,sizeof obuf,"%d",O); snprintf(sbuf,sizeof sbuf,"%d",S);
    while(*p){
        if(!strncmp(p,"I_DIM",5)){ memcpy(out+j,ibuf,strlen(ibuf)); j+=strlen(ibuf); p+=5; }
        else if(!strncmp(p,"O_DIM",5)){ memcpy(out+j,obuf,strlen(obuf)); j+=strlen(obuf); p+=5; }
        else if(!strncmp(p,"S_DIM",5)){ memcpy(out+j,sbuf,strlen(sbuf)); j+=strlen(sbuf); p+=5; }
        else if(!strncmp(p,"N_DIM",5)){ memcpy(out+j,ibuf,strlen(ibuf)); j+=strlen(ibuf); p+=5; } /* silu uses N_DIM=I */
        else { out[j++]=*p++; }
    }
    out[j]=0; return out;
}

static VkShaderModule load_module(const char* glsl){
    static int counter=0;
    char tmpl[256]; snprintf(tmpl,sizeof tmpl,"/tmp/coli_vk_%d_%d.spv",(int)getpid(),counter++);
    char glslpath[300]; snprintf(glslpath,sizeof glslpath,"%s.comp",tmpl);
    FILE* f=fopen(glslpath,"w"); if(!f){ fprintf(stderr,"[VK] fopen glsl\n"); return NULL; } fputs(glsl,f); fclose(f);
    char cmd[512]; snprintf(cmd,sizeof cmd,"glslangValidator -V %s -o %s >/dev/null 2>&1",glslpath,tmpl);
    if(system(cmd)!=0){ fprintf(stderr,"[VK] glslang failed for %s\n",glslpath); return NULL; }
    FILE* sp=fopen(tmpl,"rb"); if(!sp){ fprintf(stderr,"[VK] spv missing\n"); return NULL; }
    fseek(sp,0,SEEK_END); long sz=ftell(sp); fseek(sp,0,SEEK_SET);
    char* bin=malloc(sz); if(fread(bin,1,sz,sp)!=(size_t)sz){ fprintf(stderr,"[VK] spv read\n"); return NULL; } fclose(sp);
    VkShaderModule m; VkShaderModuleCreateInfo ci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sz,.pCode=(void*)bin};
    if(vkCreateShaderModule(g_dev,&ci,NULL,&m)!=VK_SUCCESS){ fprintf(stderr,"[VK] shaderModule\n"); return NULL; }
    free(bin); return m;
}

/* ---- helpers ---- */
static uint32_t host_visible_type(){
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(g_pd,&mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++)
        if(mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) return i;
    return 0;
}
static VkBuffer make_buf(size_t bytes, VkDeviceMemory* mem){
    VkBuffer buf; VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bytes,
        .usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    if(vkCreateBuffer(g_dev,&bci,NULL,&buf)!=VK_SUCCESS) return NULL;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev,buf,&mr);
    VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=mr.size,.memoryTypeIndex=host_visible_type()};
    if(vkAllocateMemory(g_dev,&ai,NULL,mem)!=VK_SUCCESS){ vkDestroyBuffer(g_dev,buf,NULL); return NULL; }
    vkBindBufferMemory(g_dev,buf,*mem,0);
    return buf;
}

/* Persistent scratch buffers for attention: allocated once, grown only when a
 * larger shape is needed, and kept mapped. Avoids the per-call vkCreateBuffer /
 * vkAllocateMemory / map / vkDestroyBuffer storm that forced a queue flush and
 * turned every attention layer into a ~130ms submit+wait round-trip. On this APU
 * the memory is unified, so host-visible scratch is the fast path. */
typedef struct { VkBuffer buf; VkDeviceMemory mem; size_t cap; void* map; } PersistBuf;
static PersistBuf g_pb[6]={0};   /* 0=q 1=l 2=r 3=c 4=o 5=kvbb */
static VkBuffer pb_get(int i, size_t bytes, VkDeviceMemory* mem, void** map){
    PersistBuf* p=&g_pb[i];
    if(!p->buf || bytes>p->cap){
        if(p->buf){ vkDestroyBuffer(g_dev,p->buf,NULL); vkFreeMemory(g_dev,p->mem,NULL); p->map=NULL; }
        p->buf=make_buf(bytes,&p->mem); if(!p->buf) return NULL;
        p->cap=bytes;
        if(vkMapMemory(g_dev,p->mem,0,bytes,0,&p->map)!=VK_SUCCESS){ p->map=NULL; return NULL; }
    }
    *mem=p->mem; *map=p->map; return p->buf;
}
static void pb_reset(void){ for(int i=0;i<6;i++){ if(g_pb[i].buf){ vkDestroyBuffer(g_dev,g_pb[i].buf,NULL); vkFreeMemory(g_dev,g_pb[i].mem,NULL);} g_pb[i].buf=NULL; g_pb[i].mem=NULL; g_pb[i].cap=0; g_pb[i].map=NULL; } }
static VkShaderModule g_mod_matmul=NULL, g_mod_silu=NULL;

/* ---- shader module cache (avoid recompiling glslang per call) ---- */
#define VK_CACHE_MAX 256
typedef struct { char key[32]; VkShaderModule mod; } VkModCache;
static VkModCache g_matmul_cache[VK_CACHE_MAX]; static int g_matmul_n=0;
static VkModCache g_silu_cache[VK_CACHE_MAX];   static int g_silu_n=0;

static VkShaderModule cache_get(VkModCache* c, int* n, const char* key){
    for(int i=0;i<*n;i++) if(!strcmp(c[i].key,key)) return c[i].mod;
    return NULL;
}
static VkShaderModule cache_put(VkModCache* c, int* n, const char* key, VkShaderModule mod){
    if(*n<VK_CACHE_MAX){ strncpy(c[*n].key,key,sizeof c[*n].key-1); c[*n].key[sizeof c[*n].key-1]=0; c[*n].mod=mod; (*n)++; }
    return mod;
}
static VkShaderModule get_matmul_module(int I,int O,int S){
    char key[32]; snprintf(key,sizeof key,"%d_%d_%d",I,O,S);
    VkShaderModule m=cache_get(g_matmul_cache,&g_matmul_n,key);
    if(m) return m;
    char* glsl=patch_dims(S_MATMUL,I,O,S);
    m=load_module(glsl); free(glsl);
    if(m) return cache_put(g_matmul_cache,&g_matmul_n,key,m);
    return NULL;
}
static VkShaderModule get_silu_module(int N){
    char key[32]; snprintf(key,sizeof key,"%d",N);
    VkShaderModule m=cache_get(g_silu_cache,&g_silu_n,key);
    if(m) return m;
    char* glsl=patch_dims(S_SILU,N,0,0);
    m=load_module(glsl); free(glsl);
    if(m) return cache_put(g_silu_cache,&g_silu_n,key,m);
    return NULL;
}

/* dispatch a compute shader (single bind set of 4 storage buffers) */
static int dispatch4(VkShaderModule mod, VkBuffer b0,VkBuffer b1,VkBuffer b2,VkBuffer b3,
                     uint32_t x,uint32_t y, int I,int O,int S){
    VkDescriptorSetLayoutBinding bnd[4]={
        {0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0}};
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(g_dev,&(VkDescriptorSetLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=4,.pBindings=bnd},NULL,&dsl);
    VkPipelineLayout pl; vkCreatePipelineLayout(g_dev,&(VkPipelineLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl},NULL,&pl);
    VkPipeline ppl; VkPipelineShaderStageCreateInfo ssi={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=mod,.pName="main"};
    vkCreateComputePipelines(g_dev,0,1,&(VkComputePipelineCreateInfo){.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage=ssi,.layout=pl},NULL,&ppl);
    VkDescriptorPool dp; vkCreateDescriptorPool(g_dev,&(VkDescriptorPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}},NULL,&dp);
    VkDescriptorSet ds; vkAllocateDescriptorSets(g_dev,&(VkDescriptorSetAllocateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp,.descriptorSetCount=1,.pSetLayouts=&dsl},&ds);
    VkDescriptorBufferInfo bi[4]={{b0,0,VK_WHOLE_SIZE},{b1,0,VK_WHOLE_SIZE},{b2,0,VK_WHOLE_SIZE},{b3,0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wds[4];
    for(int i=0;i<4;i++) wds[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[i]};
    vkUpdateDescriptorSets(g_dev,4,wds,0,NULL);
    VkCommandBuffer cb; vkAllocateCommandBuffers(g_dev,&(VkCommandBufferAllocateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=g_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1},&cb);
    vkBeginCommandBuffer(cb,&(VkCommandBufferBeginInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO});
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,ppl);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,0);
    vkCmdDispatch(cb,(x+TM_TILE_O-1)/TM_TILE_O,(y>0?(y+TM_TILE_S-1)/TM_TILE_S:1),1);
    vkEndCommandBuffer(cb);
    vkResetFences(g_dev,1,&g_fence);
    vkQueueSubmit(g_queue,1,&(VkSubmitInfo){.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb},g_fence);
    vkWaitForFences(g_dev,1,&g_fence,VK_TRUE,~0ULL);
    vkDestroyDescriptorPool(g_dev,dp,NULL); vkDestroyPipeline(g_dev,ppl,NULL);
    vkDestroyPipelineLayout(g_dev,pl,NULL); vkDestroyDescriptorSetLayout(g_dev,dsl,NULL);
    vkFreeCommandBuffers(g_dev,g_pool,1,&cb);
    return 1;
}

/* forward decl: convert a pipe-scratch mapped pointer to its VkBuffer handle */
/* dispatch the silu shader (2 buffers: gbuf read_write at 0, ubuf readonly at 1) */
static int dispatch2(VkShaderModule mod, VkBuffer gbuf, VkBuffer ubuf, int N){
    VkDescriptorSetLayoutBinding bnd[2]={
        {0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0}};
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(g_dev,&(VkDescriptorSetLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=2,.pBindings=bnd},NULL,&dsl);
    VkPipelineLayout pl; vkCreatePipelineLayout(g_dev,&(VkPipelineLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl},NULL,&pl);
    VkPipeline ppl; VkPipelineShaderStageCreateInfo ssi={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=mod,.pName="main"};
    vkCreateComputePipelines(g_dev,0,1,&(VkComputePipelineCreateInfo){.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage=ssi,.layout=pl},NULL,&ppl);
    VkDescriptorPool dp; vkCreateDescriptorPool(g_dev,&(VkDescriptorPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}},NULL,&dp);
    VkDescriptorSet ds; vkAllocateDescriptorSets(g_dev,&(VkDescriptorSetAllocateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp,.descriptorSetCount=1,.pSetLayouts=&dsl},&ds);
    VkDescriptorBufferInfo bi[2]={{gbuf,0,VK_WHOLE_SIZE},{ubuf,0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wds[2]={ {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[0]},
                                  {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=1,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[1]} };
    vkUpdateDescriptorSets(g_dev,2,wds,0,NULL);
    VkCommandBuffer cb; vkAllocateCommandBuffers(g_dev,&(VkCommandBufferAllocateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=g_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1},&cb);
    vkBeginCommandBuffer(cb,&(VkCommandBufferBeginInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO});
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,ppl); vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,0);
    vkCmdDispatch(cb,((uint32_t)N+63)/64,1,1);
    vkEndCommandBuffer(cb); vkResetFences(g_dev,1,&g_fence);
    vkQueueSubmit(g_queue,1,&(VkSubmitInfo){.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb},g_fence);
    vkWaitForFences(g_dev,1,&g_fence,VK_TRUE,~0ULL);
    vkDestroyDescriptorPool(g_dev,dp,NULL); vkDestroyPipeline(g_dev,ppl,NULL); vkDestroyPipelineLayout(g_dev,pl,NULL); vkDestroyDescriptorSetLayout(g_dev,dsl,NULL); vkFreeCommandBuffers(g_dev,g_pool,1,&cb);
    return 1;
}

/* Record-only variants: build pipeline + descriptor set, record the dispatch into
 * an EXISTING command buffer, and stash the handles for later cleanup. This lets
 * expert_group pack a whole layer's experts into ONE command buffer (one submit +
 * one fence wait per layer instead of one per expert). */
typedef struct { VkPipeline ppl; VkPipelineLayout pl; VkDescriptorSetLayout dsl; VkDescriptorPool dp; } RecCleanup;

static void record4(VkCommandBuffer cb, VkShaderModule mod, VkBuffer b0,VkBuffer b1,VkBuffer b2,VkBuffer b3,
                    uint32_t x,uint32_t y, size_t off_in, size_t off_out, RecCleanup* cl){
    VkDescriptorSetLayoutBinding bnd[4]={
        {0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0}};
    vkCreateDescriptorSetLayout(g_dev,&(VkDescriptorSetLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=4,.pBindings=bnd},NULL,&cl->dsl);
    vkCreatePipelineLayout(g_dev,&(VkPipelineLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&cl->dsl},NULL,&cl->pl);
    VkPipelineShaderStageCreateInfo ssi={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=mod,.pName="main"};
    vkCreateComputePipelines(g_dev,0,1,&(VkComputePipelineCreateInfo){.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage=ssi,.layout=cl->pl},NULL,&cl->ppl);
    vkCreateDescriptorPool(g_dev,&(VkDescriptorPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}},NULL,&cl->dp);
    VkDescriptorSet ds; vkAllocateDescriptorSets(g_dev,&(VkDescriptorSetAllocateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=cl->dp,.descriptorSetCount=1,.pSetLayouts=&cl->dsl},&ds);
    VkDescriptorBufferInfo bi[4]={{b0,0,VK_WHOLE_SIZE},{b1,0,VK_WHOLE_SIZE},{b2,off_in,VK_WHOLE_SIZE},{b3,off_out,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wds[4];
    for(int i=0;i<4;i++) wds[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[i]};
    vkUpdateDescriptorSets(g_dev,4,wds,0,NULL);
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,cl->ppl);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,cl->pl,0,1,&ds,0,0);
    vkCmdDispatch(cb,(x+63)/64,(y>0?y:1),1);
}
static void record2(VkCommandBuffer cb, VkShaderModule mod, VkBuffer gbuf, VkBuffer ubuf, int N, RecCleanup* cl){
    VkDescriptorSetLayoutBinding bnd[2]={
        {0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0}};
    vkCreateDescriptorSetLayout(g_dev,&(VkDescriptorSetLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=2,.pBindings=bnd},NULL,&cl->dsl);
    vkCreatePipelineLayout(g_dev,&(VkPipelineLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&cl->dsl},NULL,&cl->pl);
    VkPipelineShaderStageCreateInfo ssi={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=mod,.pName="main"};
    vkCreateComputePipelines(g_dev,0,1,&(VkComputePipelineCreateInfo){.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage=ssi,.layout=cl->pl},NULL,&cl->ppl);
    vkCreateDescriptorPool(g_dev,&(VkDescriptorPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}},NULL,&cl->dp);
    VkDescriptorSet ds; vkAllocateDescriptorSets(g_dev,&(VkDescriptorSetAllocateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=cl->dp,.descriptorSetCount=1,.pSetLayouts=&cl->dsl},&ds);
    VkDescriptorBufferInfo bi[2]={{gbuf,0,VK_WHOLE_SIZE},{ubuf,0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wds[2]={ {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[0]},
                                  {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=1,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[1]} };
    vkUpdateDescriptorSets(g_dev,2,wds,0,NULL);
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,cl->ppl);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,cl->pl,0,1,&ds,0,0);
    vkCmdDispatch(cb,((uint32_t)N+63)/64,1,1);
}
static void cleanup_rec(RecCleanup* cl){
    vkDestroyDescriptorPool(g_dev,cl->dp,NULL); vkDestroyPipeline(g_dev,cl->ppl,NULL);
    vkDestroyPipelineLayout(g_dev,cl->pl,NULL); vkDestroyDescriptorSetLayout(g_dev,cl->dsl,NULL);
}

/* ==================== ABI ==================== */

int coli_cuda_init(const int *devices, int count){
    if(g_enabled) return 1;
    VkApplicationInfo ai; memset(&ai,0,sizeof ai); ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici; memset(&ici,0,sizeof ici); ici.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo=&ai;
    if(vkCreateInstance(&ici,NULL,&g_inst)!=VK_SUCCESS){ fprintf(stderr,"[VK] instance\n"); return 0; }
    uint32_t ndev=0; vkEnumeratePhysicalDevices(g_inst,&ndev,NULL);
    if(ndev==0){ fprintf(stderr,"[VK] no devices\n"); return 0; }
    VkPhysicalDevice* pds=malloc(ndev*sizeof(VkPhysicalDevice)); vkEnumeratePhysicalDevices(g_inst,&ndev,pds);
    /* pick: prefer the device asked for, else first discrete/integrated */
    g_pd = (count>0)? NULL : pds[0];
    if(count>0){ for(uint32_t i=0;i<ndev;i++){ VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i],&p);
        if((int)p.deviceID==devices[0]) g_pd=pds[i]; } if(!g_pd) g_pd=pds[0]; }
    g_devices[0]=0; g_ndev=1; (void)devices;(void)count;
    /* queue */
    uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd,&nq,NULL);
    VkQueueFamilyProperties* qp=malloc(nq*sizeof*qp); vkGetPhysicalDeviceQueueFamilyProperties(g_pd,&nq,qp);
    g_qf=0; for(uint32_t i=0;i<nq;i++) if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){ g_qf=i; break; }
    free(qp);
    float qp_=1.0f; VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=g_qf,.queueCount=1,.pQueuePriorities=&qp_};
    if(vkCreateDevice(g_pd,&(VkDeviceCreateInfo){.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci},NULL,&g_dev)!=VK_SUCCESS){ fprintf(stderr,"[VK] device\n"); return 0; }
    vkGetDeviceQueue(g_dev,g_qf,0,&g_queue);
    vkCreateCommandPool(g_dev,&(VkCommandPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,.queueFamilyIndex=g_qf},NULL,&g_pool);
    vkCreateFence(g_dev,&(VkFenceCreateInfo){.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},NULL,&g_fence);
    /* shaders are compiled lazily + cached per shape (get_matmul_module/get_silu_module) */
    g_enabled=1; free(pds);
    fprintf(stderr,"[VK] backend ready on Vulkan device\n");
    return 1;
}

void coli_cuda_shutdown(void){
    if(!g_enabled) return;
    /* destroy cached modules */
    for(int i=0;i<g_matmul_n;i++) vkDestroyShaderModule(g_dev,g_matmul_cache[i].mod,NULL);
    for(int i=0;i<g_silu_n;i++) vkDestroyShaderModule(g_dev,g_silu_cache[i].mod,NULL);
    vkDestroyFence(g_dev,g_fence,NULL); vkDestroyCommandPool(g_dev,g_pool,NULL);
    vkDestroyDevice(g_dev,NULL); vkDestroyInstance(g_inst,NULL);
    g_enabled=0; g_inst=NULL; g_dev=NULL;
}

int coli_cuda_device_count(void){ return g_enabled?1:0; }
int coli_cuda_device_at(int index){ return index==0?0:-1; }
int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes){
    if(!g_enabled) return 0;
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(g_pd,&mp);
    VkDeviceSize biggest=0; for(uint32_t i=0;i<mp.memoryHeapCount;i++) if(mp.memoryHeaps[i].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) biggest=mp.memoryHeaps[i].size;
    if(total_bytes)*total_bytes=(size_t)biggest; if(free_bytes)*free_bytes=(size_t)biggest; /* unified mem: report all */
    (void)device; return 1;
}
void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes){
    (void)device; if(tensor_count)*tensor_count=0; if(tensor_bytes)*tensor_bytes=0;
}
void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows, double *h2d_ms, double *kernel_ms, double *d2h_ms){
    if(calls)*calls=g_calls; if(experts)*experts=g_experts; if(rows)*rows=g_rows;
    if(h2d_ms)*h2d_ms=g_h2d_ms; if(kernel_ms)*kernel_ms=g_kernel_ms; if(d2h_ms)*d2h_ms=g_d2h_ms;
}

int coli_cuda_tensor_upload(ColiCudaTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int device){
    if(!g_enabled) return 0;
    if(!*tensor){
        ColiCudaTensor* t=calloc(1,sizeof*t);
        t->device=device; t->fmt=fmt; t->I=I; t->O=O;
        size_t wbytes = fmt==0 ? (size_t)O*I*sizeof(float)
                      : fmt==1 ? (size_t)O*I*sizeof(int8_t)
                      : fmt==2 ? (size_t)O*((I+1)/2)
                      : (size_t)O*((I+3)/4);
        t->wbytes=wbytes; t->sbytes=(size_t)O*sizeof(float);
        t->wbuf=make_buf(wbytes,&t->wmem); t->sbuf=make_buf(t->sbytes,&t->smem);
        if(!t->wbuf||!t->sbuf){ fprintf(stderr,"[VK] tensor alloc\n"); return 0; }
        *tensor=t;
    }
    ColiCudaTensor* t=*tensor;
    void* p;
    vkMapMemory(g_dev,t->wmem,0,t->wbytes,0,&p); memcpy(p,weights,t->wbytes); vkUnmapMemory(g_dev,t->wmem);
    vkMapMemory(g_dev,t->smem,0,t->sbytes,0,&p); memcpy(p,scales,t->sbytes); vkUnmapMemory(g_dev,t->smem);
    return 1;
}

/* int4 matmul: y[S,O] = x[S,I] @ W[O,I]^T, W int4 (2 nibbles/byte, signed), scale per row. */
int coli_cuda_matmul(ColiCudaTensor **tensor, float *y, const float *x,
                     const void *weights, const float *scales, int fmt, int S, int I, int O, int device){
    if(fmt!=2) return 0;                 /* only int4 accelerated; else CPU */
    if(S<1) return 0;
    if(!coli_cuda_tensor_upload(tensor,weights,scales,fmt,I,O,device)) return 0;
    ColiCudaTensor* t=*tensor;
    size_t xb=(size_t)S*I*sizeof(float), yb=(size_t)S*O*sizeof(float);
    VkBuffer xb_buf, yb_buf; VkDeviceMemory xm, ym;
    xb_buf=make_buf(xb,&xm); yb_buf=make_buf(yb,&ym);
    if(!xb_buf||!yb_buf) return 0;
    void* p;
    vkMapMemory(g_dev,xm,0,xb,0,&p); memcpy(p,x,xb); vkUnmapMemory(g_dev,xm);
    g_calls++;
    dispatch4(get_matmul_module(I,O,S), t->wbuf, t->sbuf, xb_buf, yb_buf, (uint32_t)O, (uint32_t)S, I,O,S);
    vkMapMemory(g_dev,ym,0,yb,0,&p); memcpy(y,p,yb); vkUnmapMemory(g_dev,ym);
    vkDestroyBuffer(g_dev,xb_buf,NULL); vkFreeMemory(g_dev,xm,NULL);
    vkDestroyBuffer(g_dev,yb_buf,NULL); vkFreeMemory(g_dev,ym,NULL);
    return 1;
}

int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S){
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=2||up->fmt!=2||down->fmt!=2) return 0;
    if(gate->device!=up->device||gate->device!=down->device) return 0;
    int D=gate->I, I=gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float), yb=(size_t)S*D*sizeof(float);
    VkBuffer xbuf, gbuf, ubuf, ybuf; VkDeviceMemory xm,gm,um,ym;
    xbuf=make_buf(xb,&xm); gbuf=make_buf(ib,&gm); ubuf=make_buf(ib,&um); ybuf=make_buf(yb,&ym);
    if(!xbuf||!gbuf||!ubuf||!ybuf) return 0;
    void* p; vkMapMemory(g_dev,xm,0,xb,0,&p); memcpy(p,x,xb); vkUnmapMemory(g_dev,xm);
    g_experts++; g_rows+=S;
    /* gate = x @ Wg^T ; up = x @ Wu^T */
    VkShaderModule m_gate=get_matmul_module(D,I,S);
    VkShaderModule m_up  =get_matmul_module(D,I,S);
    if(!m_gate||!m_up) return 0;
    dispatch4(m_gate, gate->wbuf, gate->sbuf, xbuf, gbuf, (uint32_t)I, (uint32_t)S, D,I,S);
    dispatch4(m_up,   up->wbuf,   up->sbuf,   xbuf, ubuf, (uint32_t)I, (uint32_t)S, D,I,S);
    /* silu(gate)*up -> gbuf */
    VkShaderModule m_silu=get_silu_module((int)((size_t)S*(size_t)I));
    if(!m_silu) return 0;
    dispatch2(m_silu, gbuf, ubuf, (int)((size_t)S*(size_t)I));
    /* down = silu_gate @ Wd^T -> y */
    VkShaderModule m_down=get_matmul_module(I,D,S);
    if(!m_down) return 0;
    dispatch4(m_down, down->wbuf, down->sbuf, gbuf, ybuf, (uint32_t)D, (uint32_t)S, I,D,S);
    vkMapMemory(g_dev,ym,0,yb,0,&p); memcpy(y,p,yb); vkUnmapMemory(g_dev,ym);
    vkDestroyBuffer(g_dev,xbuf,NULL); vkFreeMemory(g_dev,xm,NULL);
    vkDestroyBuffer(g_dev,gbuf,NULL); vkFreeMemory(g_dev,gm,NULL);
    vkDestroyBuffer(g_dev,ubuf,NULL); vkFreeMemory(g_dev,um,NULL);
    vkDestroyBuffer(g_dev,ybuf,NULL); vkFreeMemory(g_dev,ym,NULL);
    return 1;
}

int coli_cuda_expert_group(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups, ColiCudaTensor *const *downs, const int *rows, int count, float *y, const float *x){
    if(!gates||!ups||!downs||!rows||!x||!y||count<1) return 0;
    ColiCudaTensor *first=gates[0];
    if(!first) return 0;
    int device=first->device, D=first->I, I=first->O, total=0, all_s4=1;
    if(count>64) return 0;  /* matches CUDA GroupDesc[64] cap */
    int offs[64];
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        all_s4&= (g->fmt==2&&u->fmt==2&&d->fmt==2);
        offs[c]=total; total+=rows[c];
    }
    if(!all_s4) return 0;  /* only int4 (fmt=2) path implemented */
    g_experts+=count; g_rows+=total;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float), yb=(size_t)total*D*sizeof(float);
    VkBuffer xbuf,gbuf,ubuf,ybuf; VkDeviceMemory xm,gm,um,ym;
    xbuf=make_buf(xb,&xm); gbuf=make_buf(ib,&gm); ubuf=make_buf(ib,&um); ybuf=make_buf(yb,&ym);
    if(!xbuf||!gbuf||!ubuf||!ybuf) return 0;
    void* p;
    vkMapMemory(g_dev,xm,0,xb,0,&p); memcpy(p,x,xb); vkUnmapMemory(g_dev,xm);
    /* x is in host-visible coherent memory -> already visible to GPU after memcpy. */
    /* NOTE: weights are already device-resident (uploaded at tensor_upload). */
    /* Record the whole layer into ONE command buffer. */
    VkShaderModule m_mm=get_matmul_module(D,I,total);   /* fmt=2 matmul, row-len I */
    VkShaderModule m_silu=get_silu_module((int)((size_t)total*(size_t)I));
    if(!m_mm||!m_silu) return 0;
    int nrec=count*3+1;  /* gate+up+down per expert, +1 silu */
    RecCleanup *rc=malloc(sizeof(RecCleanup)*nrec);
    VkCommandBuffer cb; vkAllocateCommandBuffers(g_dev,&(VkCommandBufferAllocateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=g_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1},&cb);
    vkBeginCommandBuffer(cb,&(VkCommandBufferBeginInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO});
    int ri=0;
    for(int c=0;c<count;c++){
        int r=rows[c], o=offs[c];
        /* gate: x[o*D..] -> gate[o*I..] using gates[c] weights */
        record4(cb,m_mm, gates[c]->wbuf, gates[c]->sbuf, xbuf, gbuf, (uint32_t)I, (uint32_t)r,
                (size_t)o*(size_t)D*4, (size_t)o*(size_t)I*4, &rc[ri++]);
        /* up:   x[o*D..] -> up[o*I..]   using ups[c] weights   */
        record4(cb,m_mm, ups[c]->wbuf,   ups[c]->sbuf,   xbuf, ubuf, (uint32_t)I, (uint32_t)r,
                (size_t)o*(size_t)D*4, (size_t)o*(size_t)I*4, &rc[ri++]);
    }
    /* silu(gate)*up -> gate (over full hidden, no offset) */
    record2(cb,m_silu, gbuf, ubuf, (int)((size_t)total*(size_t)I), &rc[ri++]);
    for(int c=0;c<count;c++){
        int r=rows[c], o=offs[c];
        /* down: gate[o*I..] -> y[o*D..] using downs[c] weights */
        record4(cb,m_mm, downs[c]->wbuf, downs[c]->sbuf, gbuf, ybuf, (uint32_t)D, (uint32_t)r,
                (size_t)o*(size_t)I*4, (size_t)o*(size_t)D*4, &rc[ri++]);
    }
    vkEndCommandBuffer(cb);
    vkResetFences(g_dev,1,&g_fence);
    vkQueueSubmit(g_queue,1,&(VkSubmitInfo){.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb},g_fence);
    vkWaitForFences(g_dev,1,&g_fence,VK_TRUE,~0ULL);
    vkFreeCommandBuffers(g_dev,g_pool,1,&cb);
    for(int i=0;i<nrec;i++) cleanup_rec(&rc[i]);
    free(rc);
    /* download y (one D2H) */
    vkMapMemory(g_dev,ym,0,yb,0,&p); memcpy(y,p,yb); vkUnmapMemory(g_dev,ym);
    vkDestroyBuffer(g_dev,xbuf,NULL); vkFreeMemory(g_dev,xm,NULL);
    vkDestroyBuffer(g_dev,gbuf,NULL); vkFreeMemory(g_dev,gm,NULL);
    vkDestroyBuffer(g_dev,ubuf,NULL); vkFreeMemory(g_dev,um,NULL);
    vkDestroyBuffer(g_dev,ybuf,NULL); vkFreeMemory(g_dev,ym,NULL);
    return 1;
}

void coli_cuda_tensor_free(ColiCudaTensor *tensor){
    if(!tensor) return;
    if(tensor->wbuf){ vkDestroyBuffer(g_dev,tensor->wbuf,NULL); vkFreeMemory(g_dev,tensor->wmem,NULL); }
    if(tensor->sbuf){ vkDestroyBuffer(g_dev,tensor->sbuf,NULL); vkFreeMemory(g_dev,tensor->smem,NULL); }
    free(tensor);
}
size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor){ return tensor? tensor->wbytes+tensor->sbytes : 0; }
int coli_cuda_tensor_device(const ColiCudaTensor *tensor){ return tensor? tensor->device : -1; }
int coli_cuda_tensor_update(ColiCudaTensor *tensor, const void *weights, const float *scales){
    if(!tensor) return 0;
    void* p; vkMapMemory(g_dev,tensor->wmem,0,tensor->wbytes,0,&p); memcpy(p,weights,tensor->wbytes); vkUnmapMemory(g_dev,tensor->wmem);
    vkMapMemory(g_dev,tensor->smem,0,tensor->sbytes,0,&p); memcpy(p,scales,tensor->sbytes); vkUnmapMemory(g_dev,tensor->smem);
    return 1;
}

/* ---- MLA attention weight-absorption (decode/prefill) on Vulkan ----
 * Faithful port of backend_cuda.cu's attention_absorb(_batch/_project_batch).
 * kv_b is [H*(Q+V), K] int4; rows h*(Q+V)+d (d<Q) are the q-navigation weights,
 * rows h*(Q+V)+Q+v (v<V) are the value weights. q is [S,H,Q+R]; latent [T,K];
 * rope [T,R]. score(t)= qa·latent[t] + q_rope·rope[t]; softmax over t (causal
 * nt=T-S+s+1); ctx[h*V+v]= Σ_t score(t) * (latent[t]·Wvalue[h,v]). o_proj
 * [D,H*V] maps ctx -> out.
 *
 * Push-constant layout (shared by both shaders), 48 bytes:
 *   S H Q R V K T  O  I  pad  scale
 *   i i i i i i i  i  i  i   f
 * O = out dim (D) for o_proj, unused in absorb. I = weight row len in bytes'
 * (K+1)/2 for absorb, (H*V+1)/2 for o_proj. */
/* S_ATTN_R: PATH-2 reconstruction. kvb[t, r] = sc[r] * (Lc[t,:] . W[r,:]), i.e. a
 * matmul Y[T,kvb_dim] = Lc[T,K] @ W[kvb_dim,K]^T with per-row scale sc[r]. The naive
 * kernel strided K/2 bytes apart per weight row (same pathology as the old matmul).
 * Tiled shared-memory GEMM: one workgroup computes a tileR x tileT block of kvb, loading
 * dequantized W-tile and Lc-tile into shared memory cooperatively. */
#define TR_TILE_R 8
#define TR_TILE_T 8
#define TR_TILE_K 32
static const char* S_ATTN_R =
"#version 450\n"
"layout(local_size_x=" XSTR(TR_TILE_R) ", local_size_y=" XSTR(TR_TILE_T) ") in;\n"
"layout(std430,binding=0) readonly buffer Wbuf { uint w[]; };\n"
"layout(std430,binding=1) readonly buffer Sbuf { float sc[]; };\n"
"layout(std430,binding=2) readonly buffer Lbuf { float lat[]; };\n"
"layout(std430,binding=3) writeonly buffer KVbuf { float kvb[]; };\n"
"layout(push_constant) uniform PC { int S; int H; int Q; int R; int V; int K; int T; int O; int I; int pad; float scale; } pc;\n"
"shared float smW[" XSTR(TR_TILE_R) "*" XSTR(TR_TILE_K) "];\n"
"shared float smL[" XSTR(TR_TILE_T) "*" XSTR(TR_TILE_K) "];\n"
"int wgt(uint row,uint k){ uint rb=uint(pc.I); uint bo=row*rb+(k>>1u); uint byte=(w[bo>>2u] >> ((bo&3u)*8u)) & 0xFFu; int nib=(k&1u)!=0?int(byte>>4u):int(byte&0xFu); return nib-8; }\n"
"void main(){\n"
"  uint r0=(gl_WorkGroupID.x*" XSTR(TR_TILE_R) ")+gl_LocalInvocationID.x;\n"
"  uint t0=(gl_WorkGroupID.y*" XSTR(TR_TILE_T) ")+gl_LocalInvocationID.y;\n"
"  uint lx=gl_LocalInvocationID.x, ly=gl_LocalInvocationID.y;\n"
"  uint K=uint(pc.K), R=uint(pc.H*(pc.Q+pc.V)), T=uint(pc.T);\n"
"  float acc=0.0f;\n"
"  for(uint k0=0u;k0<K;k0+=" XSTR(TR_TILE_K) "){\n"
"    for(uint e=lx+ly*" XSTR(TR_TILE_R) "; e<" XSTR(TR_TILE_R) "*" XSTR(TR_TILE_K) "; e+=" XSTR(TR_TILE_R) "*" XSTR(TR_TILE_T) "){\n"
"      uint tk=e%" XSTR(TR_TILE_K) "; uint tr=e/" XSTR(TR_TILE_K) ";\n"
"      uint gk=k0+tk; uint gr=r0-lx+tr; uint gt=t0-ly+tr;\n"
"      if(gr<R && tk<(K-k0)) smW[tr*" XSTR(TR_TILE_K) "+tk]=float(wgt(gr,gk)); else smW[tr*" XSTR(TR_TILE_K) "+tk]=0.0f;\n"
"      if(gt<T && tk<(K-k0)) smL[tr*" XSTR(TR_TILE_K) "+tk]=lat[(gt*K)+gk]; else smL[tr*" XSTR(TR_TILE_K) "+tk]=0.0f;\n"
"    }\n"
"    barrier();\n"
"    if(r0<R && t0<T){\n"
"      for(uint k=0u;k<" XSTR(TR_TILE_K) ";k++) acc += smL[ly*" XSTR(TR_TILE_K) "+k]*smW[lx*" XSTR(TR_TILE_K) "+k];\n"
"    }\n"
"    barrier();\n"
"  }\n"
"  if(r0<R && t0<T) kvb[(t0*R)+r0]=acc*sc[r0];\n"
"}\n";

/* S_ATTN_S: PATH-2 attention. One thread per (s,h,v): single-pass online
 * softmax over t (Welford-style running max/sum), so the q.kvb score is
 * computed ONCE per t instead of twice (the old two-pass kernel recomputed
 * it). All 64 threads in a workgroup share the same (s,h) and thus the same
 * q vector, so they cooperatively load q into shared memory once and read
 * it from there in the t-loop (removes per-t global re-fetches of q).
 *   score = q[k_nope].kvb[t,h*(Q+V)+0..Q-1] + q_rope.rope[t]
 *   ctx[s,h,v] = (sum_t w[t]*kvb[t, h*(Q+V)+Q+v]) / (sum_t w[t])
 *   w[t] = exp(score*scale - runningmax).  Causal: s attends t in [0,T-S+s]. */
 static const char* S_ATTN_S =
"#version 450\n"
"layout(local_size_x=64) in;\n"
"layout(std430,binding=0) readonly buffer KVbuf { float kvb[]; };\n"
"layout(std430,binding=1) readonly buffer Qbuf { float q[]; };\n"
"layout(std430,binding=2) readonly buffer Rbuf { float rope[]; };\n"
"layout(std430,binding=3) writeonly buffer Cbuf { float ctx[]; };\n"
"layout(push_constant) uniform PC { int S; int H; int Q; int R; int V; int K; int T; int O; int I; int pad; float scale; } pc;\n"
"shared float sq[256];\n"
"void main(){\n"
"  uint idx=gl_GlobalInvocationID.x;\n"
"  uint s=gl_GlobalInvocationID.y;\n"
"  if(s>=uint(pc.S)) return;\n"
"  uint h=idx/uint(pc.V); uint v=idx%uint(pc.V);\n"
"  if(h>=uint(pc.H)) return;\n"
"  int kvb_dim=pc.H*(pc.Q+pc.V);\n"
"  int rbase=int(h)*(pc.Q+pc.V);\n"
"  int nt=pc.T-pc.S+int(s)+1;\n"
"  if(nt<1) return;\n"
"  int qoff=(int(s)*pc.H+int(h))*(pc.Q+pc.R);\n"
"  int QR=pc.Q+pc.R;\n"
"  for(uint i=gl_LocalInvocationID.x;i<uint(QR);i+=64u) sq[i]=q[qoff+int(i)];\n"
"  barrier();\n"
"  float m=-1e30f, l=0.0f, acc=0.0f;\n"
"  for(int t=0;t<nt;t++){\n"
"    float a=0.0f;\n"
"    for(int k=0;k<pc.Q;k++) a += sq[k]*kvb[(uint(t)*uint(kvb_dim))+uint(rbase+k)];\n"
"    for(int d=0;d<pc.R;d++) a += sq[pc.Q+d]*rope[(uint(t)*uint(pc.R))+uint(d)];\n"
"    float sc=a*pc.scale;\n"
"    if(sc>m){ float r=exp(m-sc); l*=r; acc*=r; m=sc; }\n"
"    float w=exp(sc-m);\n"
"    l+=w;\n"
"    acc += w*kvb[(uint(t)*uint(kvb_dim))+uint(rbase+pc.Q+v)];\n"
"  }\n"
"  ctx[(uint(s)*uint(pc.H)+h)*uint(pc.V)+v]=acc/(l>0.0f?l:1.0f);\n"
"}\n";

static const char* S_ATTN_O =
"#version 450\n"
"layout(local_size_x=64) in;\n"
"layout(std430,binding=0) readonly buffer Wbuf { uint w[]; };\n"
"layout(std430,binding=1) readonly buffer Sbuf { float sc[]; };\n"
"layout(std430,binding=2) readonly buffer Cbuf { float ctx[]; };\n"
"layout(std430,binding=3) writeonly buffer Obuf { float outv[]; };\n"
"layout(push_constant) uniform PC { int S; int H; int Q; int R; int V; int K; int T; int O; int I; int pad; float scale; } pc;\n"
"int wgt(uint row,uint k){ uint rb=uint(pc.I); uint bo=row*rb+(k>>1u); uint byte=(w[bo>>2u] >> ((bo&3u)*8u)) & 0xFFu; int nib=(k&1u)!=0?int(byte>>4u):int(byte&0xFu); return nib-8; }\n"
"void main(){\n"
"  uint s=gl_GlobalInvocationID.y; uint d=gl_GlobalInvocationID.x;\n"
"  if(s>=uint(pc.S) || d>=uint(pc.O)) return;\n"
"  int HV=pc.H*pc.V; float acc=0.0f;\n"
"  for(int hv=0;hv<HV;hv++) acc += ctx[(uint(s)*uint(HV))+uint(hv)]*float(wgt(uint(d),uint(hv)))*sc[uint(d)];\n"
"  outv[uint(s)*uint(pc.O)+d]=acc;\n"
"}\n";

static VkShaderModule g_mod_attn_r=NULL, g_mod_attn_s=NULL, g_mod_attn_o=NULL;
static int g_attn_failed=0;
static int dbg_attn_init=0;

/* cached pipeline/layout per module (created once, reused across calls) */
typedef struct { VkShaderModule mod; VkPipeline ppl; VkPipelineLayout pl; VkDescriptorSetLayout dsl; } AttnPipe;
static AttnPipe g_pipe[3]={0};

/* Persistent dispatch resources: ONE command buffer + fence + per-role descriptor
 * set, reused across every layer-call so we do a single submit+wait per call
 * instead of one synced submit per dispatch. [0]=reconstruct(R), [1]=attend(S), [2]=o_proj(O). */
static VkCommandBuffer g_attn_cb=NULL;
static VkDescriptorPool g_attn_dp=NULL;
static VkDescriptorSet g_attn_ds[3]={NULL,NULL,NULL};
static int g_attn_rsrc_ok=0;
static AttnPipe* get_attn_pipe(VkShaderModule mod);

static int attn_ensure_resources(void){
    if(!g_dev) return 0;
    if(!g_attn_rsrc_ok){
        VkCommandBufferAllocateInfo cai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool=g_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
        if(vkAllocateCommandBuffers(g_dev,&cai,&g_attn_cb)!=VK_SUCCESS) return 0;
        VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,16};
        VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets=3,.poolSizeCount=1,.pPoolSizes=&ps,.flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT};
        if(vkCreateDescriptorPool(g_dev,&dpci,NULL,&g_attn_dp)!=VK_SUCCESS) return 0;
        g_attn_rsrc_ok=1;
    }
    /* Ensure each loaded module has its pipeline/layout created, then allocate the
     * matching persistent descriptor set if it is not yet allocated. Modules load
     * lazily (g_mod_attn_o only when an o_proj call happens), so this must run on
     * every call to catch a newly-loaded module whose descriptor set is still NULL. */
    if(g_mod_attn_r) get_attn_pipe(g_mod_attn_r);
    if(g_mod_attn_s) get_attn_pipe(g_mod_attn_s);
    if(g_mod_attn_o) get_attn_pipe(g_mod_attn_o);
    for(int i=0;i<3;i++){
        if(!g_attn_ds[i] && g_pipe[i].dsl){
            VkDescriptorSetAllocateInfo dai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool=g_attn_dp,.descriptorSetCount=1,.pSetLayouts=&g_pipe[i].dsl};
            if(vkAllocateDescriptorSets(g_dev,&dai,&g_attn_ds[i])!=VK_SUCCESS) return 0;
        }
    }
    return 1;
}

static AttnPipe* get_attn_pipe(VkShaderModule mod){
    for(int i=0;i<3;i++) if(g_pipe[i].mod==mod) return &g_pipe[i];
    AttnPipe* p=NULL;
    for(int i=0;i<3;i++) if(!g_pipe[i].mod){ p=&g_pipe[i]; break; }
    if(!p) return NULL;
    VkDescriptorSetLayoutBinding bnd[6]={
        {0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {4,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {5,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0}};
    vkCreateDescriptorSetLayout(g_dev,&(VkDescriptorSetLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=6,.pBindings=bnd},NULL,&p->dsl);
    VkPushConstantRange pcr={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=48};
    vkCreatePipelineLayout(g_dev,&(VkPipelineLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&p->dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr},NULL,&p->pl);
    VkPipelineShaderStageCreateInfo ssi={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=mod,.pName="main"};
    vkCreateComputePipelines(g_dev,0,1,&(VkComputePipelineCreateInfo){.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage=ssi,.layout=p->pl},NULL,&p->ppl);
    p->mod=mod;
    return p;
}

/* Record one dispatch into the persistent command buffer g_attn_cb.
 * Updates the persistent descriptor set `ds` to point at b0..b5, binds it with
 * the given push constants, and emits a vkCmdDispatch. No submit/wait here. */
static int attn_record(VkShaderModule mod, VkDescriptorSet ds, int nbind,
                        VkBuffer b0,VkBuffer b1,VkBuffer b2,VkBuffer b3,VkBuffer b4,VkBuffer b5,
                        uint32_t x,uint32_t y, int S,int H,int Q,int R,int V,int K,int T,int O,int I,float scale,
                        uint32_t lx_dim,uint32_t ly_dim){
    AttnPipe* ap=get_attn_pipe(mod);
    if(!ap||!ap->ppl||!ds) return 0;
    VkBuffer bb[6]={b0,b1,b2,b3,b4,b5};
    VkDescriptorBufferInfo bi[6]; VkWriteDescriptorSet wds[6];
    for(int i=0;i<nbind;i++){ bi[i]=(VkDescriptorBufferInfo){bb[i],0,VK_WHOLE_SIZE};
        wds[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[i]}; }
    vkUpdateDescriptorSets(g_dev,nbind,wds,0,NULL);
    int pc[10]; float pcf; pc[0]=S;pc[1]=H;pc[2]=Q;pc[3]=R;pc[4]=V;pc[5]=K;pc[6]=T;pc[7]=O;pc[8]=I;pc[9]=0;pcf=scale;
    uint8_t pcbuf[48]; memset(pcbuf,0,48); memcpy(pcbuf,pc,40); memcpy(pcbuf+40,&pcf,4);
    vkCmdBindPipeline(g_attn_cb,VK_PIPELINE_BIND_POINT_COMPUTE,ap->ppl);
    vkCmdBindDescriptorSets(g_attn_cb,VK_PIPELINE_BIND_POINT_COMPUTE,ap->pl,0,1,&ds,0,0);
    vkCmdPushConstants(g_attn_cb,ap->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,48,pcbuf);
    vkCmdDispatch(g_attn_cb,(x+lx_dim-1)/lx_dim,(y>0?(y+ly_dim-1)/ly_dim:1),1);
    return 1;
}

/* ---- pipe_* / attention: implemented for attention, stubs for pipe_* ----
 * Option B (whole-layer fused GPU pipeline) was prototyped here: rmsnorm, rope,
 * add, copy2d, silu_mul, gemm (int4) ops plus a resident command-buffer
 * fusion. On this unified-memory APU it was measurably SLOWER than the
 * attention-only path (0.09-0.11 vs 0.33 tok/s) and unstable
 * (VK_ERROR_DEVICE_LOST under barrier-chained multi-dispatch cb), so the
 * default Vulkan config keeps the per-layer attention path (COLI_CUDA_ATTN).
 * The resident-fusion machinery is retained as a reference but the ops fall
 * back to CPU (return 0) so COLI_CUDA_PIPE=2 degrades gracefully. */
float *coli_cuda_pipe_scratch(int d,int s,size_t b){(void)d;(void)s;(void)b;return NULL;}
void *coli_cuda_pipe_alloc(int d,size_t b){(void)d;(void)b;return NULL;}
void coli_cuda_pipe_free(int d,void*p){(void)d;(void)p;}
int coli_cuda_pipe_upload(int d,void*dst,const void*src,size_t b){(void)d;(void)dst;(void)src;(void)b;return 0;}
int coli_cuda_pipe_download(int d,const void*src,void*dst,size_t b){(void)d;(void)src;(void)dst;(void)b;return 0;}
int coli_cuda_pipe_rmsnorm(int d,float*y,const float*x,const float*w,int S,int D,float e){(void)d;(void)y;(void)x;(void)w;(void)S;(void)D;(void)e;return 0;}
int coli_cuda_pipe_rope(int d,float*v,const int*pos,int r,int st,int off,int R,int h,float th){(void)d;(void)v;(void)pos;(void)r;(void)st;(void)off;(void)R;(void)h;(void)th;return 0;}
int coli_cuda_pipe_silu_mul(int d,float*g,const float*u,size_t n){(void)d;(void)g;(void)u;(void)n;return 0;}
int coli_cuda_pipe_add(int d,float*x,const float*t,size_t n){(void)d;(void)x;(void)t;(void)n;return 0;}
int coli_cuda_pipe_rows_add(int d,float*x,const float*p,const int*r,int n,int D){(void)d;(void)x;(void)p;(void)r;(void)n;(void)D;return 0;}
int coli_cuda_pipe_gemm(ColiCudaTensor*t,float*y,const float*x,int S){(void)t;(void)y;(void)x;(void)S;return 0;}
int coli_cuda_pipe_rmsnorm_s(int d,float*y,const float*x,const float*w,int S,int D,float e,int xs,int ys){(void)d;(void)y;(void)x;(void)w;(void)S;(void)D;(void)e;(void)xs;(void)ys;return 0;}
int coli_cuda_pipe_rope_base(int d,float*v,int pb,int r,int st,int off,int R,int h,float th){(void)d;(void)v;(void)pb;(void)r;(void)st;(void)off;(void)R;(void)h;(void)th;return 0;}
int coli_cuda_pipe_copy2d(int d,float*dst,int dp,const float*src,int sp,int w,int h){(void)d;(void)dst;(void)dp;(void)src;(void)sp;(void)w;(void)h;return 0;}
int coli_cuda_pipe_peer_copy(int dd,float*dst,int sd,const void*src,size_t b){(void)dd;(void)dst;(void)sd;(void)src;(void)b;return 0;}
int coli_cuda_pipe_sync(int d){(void)d;return 0;}
/* forward declaration so _absorb can reuse _project_batch defined below */
int coli_cuda_attention_project_batch(ColiCudaTensor*kv,ColiCudaTensor*o,float*out,const float*q,const float*l,const float*r,int S,int H,int Q,int R,int V,int K,int T,float sc);

/* CPU reference port of glm.c attention, CAUSAL (matches the Vulkan batch kernel:
 * position s attends to t in [0, T-S+s]). Used only by COLI_ATTN_DEBUG. Reads
 * weights by copying the GPU buffer into a staging buffer (avoids remap crash). */
static void attn_cpu_ref(const ColiCudaTensor*kv,float*c,const float*q,const float*l,
                         const float*r,int S,int H,int Q,int R,int V,int K,int T,float sc){
    int rb=(K+1)/2; size_t wb=kv->wbytes, sb=kv->sbytes;
    uint8_t*w=malloc(wb); float*ws=malloc(sb);
    void*pp;
    vkMapMemory(g_dev,kv->wmem,0,wb,0,&pp); memcpy(w,pp,wb); vkUnmapMemory(g_dev,kv->wmem);
    vkMapMemory(g_dev,kv->smem,0,sb,0,&pp); memcpy(ws,pp,sb); vkUnmapMemory(g_dev,kv->smem);
    for(int s=0;s<S;s++){
        int nt=T-S+s+1;
        for(int h=0;h<H;h++){
            int rbase=h*(Q+V); int qoff=(s*H+h)*(Q+R);
            float*qa=malloc((size_t)K*sizeof(float));
            for(int k=0;k<K;k++){ float a=0; for(int d=0;d<Q;d++){
                uint8_t b=w[((size_t)rbase+d)*rb+(k>>1)]; int nib=(k&1)?b>>4:b&15;
                a+=q[qoff+d]*((float)nib-8.0f)*ws[rbase+d]; } qa[k]=a; }
            float*scv=malloc((size_t)nt*sizeof(float)); float mx=-1e30f;
            for(int t=0;t<nt;t++){ float a=0; for(int k=0;k<K;k++)a+=qa[k]*l[(size_t)t*K+k];
                for(int d=0;d<R;d++)a+=q[qoff+Q+d]*r[(size_t)t*R+d];
                scv[t]=a*sc; if(scv[t]>mx)mx=scv[t]; }
            float sum=0; for(int t=0;t<nt;t++){ scv[t]=expf(scv[t]-mx); sum+=scv[t]; }
            float inv=1.0f/sum; float*cl=malloc((size_t)K*sizeof(float));
            for(int k=0;k<K;k++){ float a=0; for(int t=0;t<nt;t++)a+=scv[t]*inv*l[(size_t)t*K+k]; cl[k]=a; }
            for(int v=0;v<V;v++){ int row=rbase+Q+v; float a=0;
                for(int k=0;k<K;k++){ uint8_t b=w[((size_t)row)*rb+(k>>1)]; int nib=(k&1)?b>>4:b&15;
                    a+=cl[k]*((float)nib-8.0f); }
                c[((size_t)s*H+h)*V+v]=a*ws[row]; }
            free(qa); free(scv); free(cl);
        }
    }
    free(w); free(ws);
}

/* PATH-2 CPU reference: reconstruct kvb = matmul_qt(Lc,kv_b) then attend with raw q.
 * Mirrors glm.c attention_rows() so we can diff the GPU R+S shaders against it. */
static void attn_cpu_ref_path2(const ColiCudaTensor*kv,float*c,const float*q,const float*l,
                               const float*r,int S,int H,int Q,int R,int V,int K,int T,float sc){
    int rb=(K+1)/2; int kvb_dim=H*(Q+V); size_t wb=kv->wbytes, sb=kv->sbytes;
    uint8_t*w=malloc(wb); float*ws=malloc(sb);
    void*pp;
    vkMapMemory(g_dev,kv->wmem,0,wb,0,&pp); memcpy(w,pp,wb); vkUnmapMemory(g_dev,kv->wmem);
    vkMapMemory(g_dev,kv->smem,0,sb,0,&pp); memcpy(ws,pp,sb); vkUnmapMemory(g_dev,kv->smem);
    float*kvb=malloc((size_t)T*kvb_dim*sizeof(float));
    for(int t=0;t<T;t++) for(int rr=0;rr<kvb_dim;rr++){
        float a=0; for(int d=0;d<K;d++){ uint8_t b=w[((size_t)rr)*rb+(d>>1)]; int nib=(d&1)?b>>4:b&15;
            a+=l[(size_t)t*K+d]*((float)nib-8.0f)*ws[rr]; }
        kvb[(size_t)t*kvb_dim+rr]=a;
    }
    for(int s=0;s<S;s++){
        int nt=T-S+s+1;
        for(int h=0;h<H;h++){
            int rbase=h*(Q+V); int qoff=(s*H+h)*(Q+R);
            float*scv=malloc((size_t)nt*sizeof(float)); float mx=-1e30f;
            for(int t=0;t<nt;t++){ float a=0;
                for(int k=0;k<Q;k++) a+=q[qoff+k]*kvb[(size_t)t*kvb_dim+rbase+k];
                for(int d=0;d<R;d++) a+=q[qoff+Q+d]*r[(size_t)t*R+d];
                scv[t]=a*sc; if(scv[t]>mx)mx=scv[t]; }
            float sum=0; for(int t=0;t<nt;t++){ scv[t]=expf(scv[t]-mx); sum+=scv[t]; }
            float inv=1.0f/(sum>0?sum:1.0f); float*cl=malloc((size_t)V*sizeof(float));
            for(int v=0;v<V;v++) cl[v]=0;
            for(int t=0;t<nt;t++){ float ww=scv[t]*inv;
                for(int v=0;v<V;v++) cl[v]+=ww*kvb[(size_t)t*kvb_dim+rbase+Q+v]; }
            for(int v=0;v<V;v++) c[((size_t)s*H+h)*V+v]=cl[v];
            free(scv); free(cl);
        }
    }
    free(w); free(ws); free(kvb);
}

int coli_cuda_attention_absorb(ColiCudaTensor*kv,float*c,const float*q,const float*l,const float*r,int H,int Q,int R,int V,int K,int T,float sc){
    if(!kv||!c||!q||!l||!r||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       kv->I!=K||kv->O!=H*(Q+V)) return 0;
    int ok=coli_cuda_attention_project_batch(kv,NULL,c,q,l,r,1,H,Q,R,V,K,T,sc);
    if(getenv("COLI_ATTN_DEBUG")){
        static int dbg=0;
        if(dbg<12){
            float*ref=malloc((size_t)H*V*sizeof(float));
            attn_cpu_ref(kv,ref,q,l,r,1,H,Q,R,V,K,T,sc);
            float md=0; for(int i=0;i<H*V;i++){ float d=fabsf(ref[i]-c[i]); if(d>md)md=d; }
            fprintf(stderr,"[ATTN-DBG] call %d T=%d H=%d V=%d K=%d Q=%d R=%d scale=%.6f maxdiff=%.6f\n",
                    dbg,T,H,V,K,Q,R,sc,md);
            free(ref); dbg++;
        }
    }
    return ok;
}
int coli_cuda_attention_absorb_batch(ColiCudaTensor*kv,float*c,const float*q,const float*l,const float*r,int S,int H,int Q,int R,int V,int K,int T,float sc){
    if(!kv||!c||!q||!l||!r||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||T<S||T>8192||
       kv->I!=K||kv->O!=H*(Q+V)) return 0;
    return coli_cuda_attention_project_batch(kv,NULL,c,q,l,r,S,H,Q,R,V,K,T,sc);
}
int coli_cuda_attention_project_batch(ColiCudaTensor*kv,ColiCudaTensor*o,float*out,const float*q,const float*l,const float*r,int S,int H,int Q,int R,int V,int K,int T,float sc){
    static int _dbg_sb=0; if(!_dbg_sb){ setvbuf(stderr,NULL,_IONBF,0); _dbg_sb=1; }
    if(!kv||!out||!q||!l||!r||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||T<S||T>8192||
       kv->I!=K||kv->O!=H*(Q+V)) return 0;
    double _t0=now_s(), _t_alloc=0,_t_sub=0,_t_wait=0,_t_free=0; int _timing=!!getenv("COLI_ATTN_TIMING");
    #define _TACC(var) do{ double _n=now_s(); var+=(_n-_t0); _t0=_n; }while(0)
    if(_timing){ static int _fc=0; if(_fc<1){ fprintf(stderr,"[ATTN-TIMING] ENTER S=%d H=%d V=%d K=%d T=%d o=%p\n",S,H,V,K,T,(void*)o); _fc++; } }
    if(o && (o->device!=kv->device || o->I!=H*V)) return 0;
    if(g_attn_failed) return 0;
    if(!g_mod_attn_r){ g_mod_attn_r=load_module(S_ATTN_R); if(!g_mod_attn_r){ g_attn_failed=1; return 0; } }
    if(!g_mod_attn_s){ g_mod_attn_s=load_module(S_ATTN_S); if(!g_mod_attn_s){ g_attn_failed=1; return 0; } }
    if(o && !g_mod_attn_o){ g_mod_attn_o=load_module(S_ATTN_O); if(!g_mod_attn_o){ g_attn_failed=1; return 0; } }
    int Iw=(K+1)/2;          /* kv_b int4 row bytes */
    int Io=(H*V+1)/2;        /* o_proj int4 row bytes */
    if(getenv("COLI_ATTN_DEBUG")&&dbg_attn_init<1){
        fprintf(stderr,"[ATTN-DBG] mod_attn_r=%p failed=%d Iw=%d Io=%d K=%d V=%d H=%d S=%d T=%d wbuf_ok=%d\n",(void*)g_mod_attn_r,g_attn_failed,Iw,Io,K,V,H,S,T,(int)(kv->wbytes>0));
        uint8_t*cpum=kv->wmem?(uint8_t*)kv->wmem:NULL;
        uint8_t*cpub=kv->wbuf?(uint8_t*)kv->wbuf:NULL;
        if(cpum) fprintf(stderr,"[ATTN-DBG] CPU wmem[192*256]=0x%02x  [0]=0x%02x\n", cpum[192*256], cpum[0]);
        if(cpub){ void*mpp; if(vkMapMemory(g_dev,kv->wmem,0,kv->wbytes,0,&mpp)==VK_SUCCESS){ uint8_t*mb=(uint8_t*)mpp; fprintf(stderr,"[ATTN-DBG] GPU wbuf[192*256]=0x%02x  [0]=0x%02x (mapped via wmem)\n", mb[192*256], mb[0]); vkUnmapMemory(g_dev,kv->wmem); } }
        dbg_attn_init++;
    }
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float), lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float), cb=(size_t)S*H*V*sizeof(float);
    size_t ob = o ? (size_t)S*o->O*sizeof(float) : cb;
    int kvb_dim=H*(Q+V);
    size_t kvbb=(size_t)T*kvb_dim*sizeof(float);   /* reconstructed k_nope||v per position */
    VkBuffer qbuf,lbuf,rbuf,cbuf,obuf,kvbbuf; VkDeviceMemory qm,lm,rm,cm,om,kvbm;
    void *qmap,*lmap,*rmap,*cmap,*omap,*kvbmap;
    qbuf=pb_get(0,qb,&qm,&qmap); lbuf=pb_get(1,lb,&lm,&lmap); rbuf=pb_get(2,rb,&rm,&rmap);
    cbuf=pb_get(3,cb,&cm,&cmap); obuf=pb_get(4,ob,&om,&omap); kvbbuf=pb_get(5,kvbb,&kvbm,&kvbmap);
    if(!qbuf||!lbuf||!rbuf||!cbuf||!obuf||!kvbbuf) return 0;
    if(_timing) _TACC(_t_alloc);
    memcpy(qmap,q,qb);
    memcpy(lmap,l,lb);
    memcpy(rmap,r,rb);
    if(!attn_ensure_resources()) return 0;
    vkResetCommandBuffer(g_attn_cb,0);
    vkBeginCommandBuffer(g_attn_cb,&(VkCommandBufferBeginInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO});
    /* PATH-2 reconstruction: kvb[t,r] = sum_d Lc[t,d]*wgt(r,d)*sc[r]  (R shader) */
    attn_record(g_mod_attn_r, g_attn_ds[0], 4, kv->wbuf, kv->sbuf, lbuf, kvbbuf, kvbbuf, kvbbuf,
                 (uint32_t)kvb_dim, (uint32_t)T, S,H,Q,R,V,K,T,0, Iw, sc, TR_TILE_R, TR_TILE_T);
    { VkBufferMemoryBarrier bmb={.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,.buffer=kvbbuf,.offset=0,.size=VK_WHOLE_SIZE,
        .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
       vkCmdPipelineBarrier(g_attn_cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,NULL, 1,&bmb, 0,NULL); }
    /* PATH-2 attend: score raw q vs kvb k_nope + rope, softmax, aggregate v  (S shader) */
    attn_record(g_mod_attn_s, g_attn_ds[1], 4, kvbbuf, qbuf, rbuf, cbuf, cbuf, cbuf,
                 (uint32_t)((int64_t)H*V), (uint32_t)S, S,H,Q,R,V,K,T,0, Iw, sc, 64, 1);
    { VkBufferMemoryBarrier bmb={.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,.buffer=cbuf,.offset=0,.size=VK_WHOLE_SIZE,
        .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
      vkCmdPipelineBarrier(g_attn_cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,NULL, 1,&bmb, 0,NULL); }
    if(o){
        attn_record(g_mod_attn_o, g_attn_ds[2], 4, o->wbuf, o->sbuf, cbuf, obuf, cbuf, obuf,
                      (uint32_t)o->O, (uint32_t)S, S,H,Q,R,V,K,T, o->O, Io, sc, 64, 1);
    }
    VkResult e1=vkEndCommandBuffer(g_attn_cb);
    if(e1!=VK_SUCCESS) fprintf(stderr,"[ATTN-DBG] vkEndCommandBuffer=%d\n",(int)e1);
    vkResetFences(g_dev,1,&g_fence);
    VkResult e2=vkQueueSubmit(g_queue,1,&(VkSubmitInfo){.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&g_attn_cb},g_fence);
    if(e2!=VK_SUCCESS) fprintf(stderr,"[ATTN-DBG] vkQueueSubmit=%d\n",(int)e2);
    if(_timing) _TACC(_t_sub);
    VkResult e3=vkWaitForFences(g_dev,1,&g_fence,VK_TRUE,~0ULL);
    if(e3!=VK_SUCCESS) fprintf(stderr,"[ATTN-DBG] vkWaitForFences=%d\n",(int)e3);
    if(_timing) _TACC(_t_wait);

    if(_timing){ static double sa=0,ss=0,sw=0,sf=0; static int nc=0;
        sa+=_t_alloc; ss+=_t_sub; sw+=_t_wait; sf+=_t_free; nc++;
        if(nc%78==0) fprintf(stderr,"[ATTN-TIMING] calls=%d total alloc=%.2fms submit=%.2fms wait=%.2fms (per-call a=%.4f s=%.4f w=%.4f)\n",
            nc, sa*1000, ss*1000, sw*1000, _t_alloc*1000, _t_sub*1000, _t_wait*1000); }
    if(o){
        memcpy(out,omap,ob);
    } else {
        memcpy(out,cmap,cb);
    }
    if(getenv("COLI_ATTN_DEBUG")){
        float*ref=malloc((size_t)S*H*V*sizeof(float));
        attn_cpu_ref_path2(kv,ref,q,l,r,S,H,Q,R,V,K,T,sc);
        const float*cc = (const float*)cmap;
        static int dbg2=0;
        if(dbg2<12){
            float md=0; int ami=0;
            for(int i=0;i<S*H*V;i++){ float d=fabsf(ref[i]-cc[i]); if(d>md){md=d;ami=i;} }
            fprintf(stderr,"[ATTN-DBG] path2 ctx call %d S=%d T=%d H=%d V=%d K=%d Q=%d R=%d scale=%.6f ctx_maxdiff=%.6f argmax_i=%d (s=%d h=%d v=%d) gpu=%.6f ref=%.6f\n",
                    dbg2,S,T,H,V,K,Q,R,sc,md,ami,ami/(H*V), (ami/V)%H, ami%(H*V)%V, cc[ami], ref[ami]);
        }
        if(getenv("COLI_ATTN_DEBUG") && getenv("COLI_ATTN_DBG_O") && o){
            static int dbg3=0;
            if(dbg3<12){
                int HV=H*V, D=o->O; float*oref=malloc((size_t)S*D*sizeof(float)); float*ws=malloc((size_t)D*sizeof(float));
                uint8_t*ow=malloc(o->wbytes); void*op;
                vkMapMemory(g_dev,o->wmem,0,o->wbytes,0,&op); memcpy(ow,op,o->wbytes); vkUnmapMemory(g_dev,o->wmem);
                vkMapMemory(g_dev,o->smem,0,o->sbytes,0,&op); memcpy(ws,op,o->sbytes); vkUnmapMemory(g_dev,o->smem);
                int rb=(HV+1)/2; const float*cc2=(const float*)cmap;
                for(int s=0;s<S;s++) for(int d=0;d<D;d++){
                    double a=0; for(int hv=0;hv<HV;hv++){
                        uint8_t b=ow[((size_t)d)*rb+(hv>>1)]; int nib=(hv&1)?b>>4:b&15;
                        a+=cc2[((size_t)s*HV)+hv]*((double)nib-8.0); }
                    oref[((size_t)s*D)+d]=(float)(a*ws[d]); }
                float md=0; const float*oo=(const float*)omap;
                for(int i=0;i<S*D;i++){ float dd=fabsf(oref[i]-oo[i]); if(dd>md)md=dd; }
                fprintf(stderr,"[ATTN-DBG] oproj call %d S=%d D=%d HV=%d oproj_maxdiff=%.6f\n", dbg3,S,D,HV,md);
                free(oref); free(ws); free(ow); dbg3++;
            }
        }
        if(getenv("COLI_ATTN_DUMP_KVB")){
            int rb=(K+1)/2; uint8_t*ww=malloc(kv->wbytes); float*ws=malloc(kv->sbytes); void*mp;
            vkMapMemory(g_dev,kv->wmem,0,kv->wbytes,0,&mp); memcpy(ww,mp,kv->wbytes); vkUnmapMemory(g_dev,kv->wmem);
            vkMapMemory(g_dev,kv->smem,0,kv->sbytes,0,&mp); memcpy(ws,mp,kv->sbytes); vkUnmapMemory(g_dev,kv->smem);
            /* download kvb device buffer to a temp staging to compare */
            size_t ksz=kvbb; VkBuffer tk; VkDeviceMemory tkm; tk=make_buf(ksz,&tkm); void* tkp;
            VkBufferCopy cp={0,0,(VkDeviceSize)ksz};
            vkResetCommandBuffer(g_attn_cb,0);
            vkBeginCommandBuffer(g_attn_cb,&(VkCommandBufferBeginInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO});
            vkCmdCopyBuffer(g_attn_cb,kvbbuf,tk,1,&cp);
            vkEndCommandBuffer(g_attn_cb); vkResetFences(g_dev,1,&g_fence);
            vkQueueSubmit(g_queue,1,&(VkSubmitInfo){.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&g_attn_cb},g_fence);
            vkWaitForFences(g_dev,1,&g_fence,VK_TRUE,~0ULL);
            vkMapMemory(g_dev,tkm,0,ksz,0,&tkp); float*kvbg=(float*)tkp;
            float kvbmd=0,intmi=0; for(int t=0;t<T;t++) for(int rr=0;rr<kvb_dim;rr++){
                float a=0; for(int d=0;d<K;d++){ uint8_t b=ww[((size_t)rr)*rb+(d>>1)]; int nib=(d&1)?b>>4:b&15; a+=l[(size_t)t*K+d]*((float)nib-8.0f)*ws[rr]; }
                float ddiff=fabsf(a-kvbg[(size_t)t*kvb_dim+rr]); if(ddiff>kvbmd){kvbmd=ddiff;intmi=(int)((size_t)t*kvb_dim+rr);} }
            fprintf(stderr,"[ATTN-DBG] R kvb_maxdiff=%.6f at idx=%d (t=%d r=%d)\n", kvbmd, intmi, intmi/kvb_dim, intmi-(intmi/kvb_dim)*kvb_dim);
            vkUnmapMemory(g_dev,tkm); vkDestroyBuffer(g_dev,tk,NULL); vkFreeMemory(g_dev,tkm,NULL);
            free(ww);free(ws);
        }
        free(ref); dbg2++;
    }
    if(_timing) _TACC(_t_free);   /* free phase now covers only the (cheap) final sync */
    return 1;
}
int coli_cuda_attention_project_batch_dev(ColiCudaTensor*kv,ColiCudaTensor*o,float*out,const float*qd,const float*ld,const float*rd,int S,int H,int Q,int R,int V,int K,int T,float s){
    /* qd/ld/rd are device-resident (unified-mem mapped) pointers; the base
     * project_batch uploads via memcpy into scratch (a no-op-cost copy on
     * unified memory) and copies the result back to `out`. Identical math. */
    return coli_cuda_attention_project_batch(kv,o,out,qd,ld,rd,S,H,Q,R,V,K,T,s);
}
int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor*kvb,float*cd,const float*qd,const float*ld,const float*rd,int S,int H,int Q,int R,int V,int K,int T,float s){
    return coli_cuda_attention_absorb_batch(kvb,cd,qd,ld,rd,S,H,Q,R,V,K,T,s);
}
int coli_cuda_attention_absorb_kvdev(ColiCudaTensor*kv,float*c,const float*q,const float*ld,const float*rd,int H,int Q,int R,int V,int K,int T,float s){
    return coli_cuda_attention_absorb(kv,c,q,ld,rd,H,Q,R,V,K,T,s);
}
int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor*kv,ColiCudaTensor*o,float*od,const float*qd,const float*ld,const float*rd,int S,int H,int Q,int R,int V,int K,int T,float s){
    return coli_cuda_attention_project_batch(kv,o,od,qd,ld,rd,S,H,Q,R,V,K,T,s);
}
int coli_cuda_shared_mlp_w4a16(ColiCudaTensor*g,ColiCudaTensor*u,ColiCudaTensor*d,float*y,const float*x,int S){
    (void)g;(void)u;(void)d;(void)y;(void)x;(void)S;return 0;
}
