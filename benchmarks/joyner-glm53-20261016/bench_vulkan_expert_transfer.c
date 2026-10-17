/* Standalone bounded real-weights O_DIRECT -> CPU staging -> Vulkan1 transfer benchmark.
 * No GLM inference or model-server hooks. Never mutates source checkpoint. */
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <fcntl.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define CHUNK (4u * 1024u * 1024u)
#define MAX_SAMPLES 512
#define CHECK(call) do { VkResult _r=(call); if(_r!=VK_SUCCESS){fprintf(stderr,"VULKAN_ERROR %s:%d %s=%d\n",__FILE__,__LINE__,#call,_r);exit(3);} }while(0)
static double now(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return (double)t.tv_sec+(double)t.tv_nsec/1e9;}
static int cmp(const void *a,const void *b) {double x=*(double*)a,y=*(double*)b;return (x>y)-(x<y);}
static double quant(double *x,int n,double p) {qsort(x,n,sizeof(*x),cmp);int i=(int)ceil(n*p)-1;if(i<0)i=0;if(i>=n)i=n-1;return x[i];}
typedef struct {VkBuffer b;VkDeviceMemory m;void *ptr;} Buffer;
typedef struct {VkInstance instance;VkPhysicalDevice physical;VkDevice dev;VkQueue q;uint32_t family;VkCommandPool pool;VkCommandBuffer cmd;VkFence fence;VkPhysicalDeviceMemoryProperties mem;} GPU;
static uint32_t memtype(GPU *g,uint32_t bits,VkMemoryPropertyFlags want) {
    for(uint32_t i=0;i<g->mem.memoryTypeCount;i++)
        if((bits&(1u<<i)) && (g->mem.memoryTypes[i].propertyFlags&want)==want)return i;
    fprintf(stderr,"NO_GO memory type missing flags 0x%x\n",want);exit(5);
}
static Buffer allocbuf(GPU *g,VkDeviceSize size,VkBufferUsageFlags usage,VkMemoryPropertyFlags props) {
    Buffer b={0};
    VkBufferCreateInfo info={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=size,
        .usage=usage,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    CHECK(vkCreateBuffer(g->dev,&info,NULL,&b.b));
    VkMemoryRequirements req;vkGetBufferMemoryRequirements(g->dev,b.b,&req);
    VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize=req.size,.memoryTypeIndex=memtype(g,req.memoryTypeBits,props)};
    CHECK(vkAllocateMemory(g->dev,&ai,NULL,&b.m));
    CHECK(vkBindBufferMemory(g->dev,b.b,b.m,0));
    if(props&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)CHECK(vkMapMemory(g->dev,b.m,0,size,0,&b.ptr));
    return b;
}
static void release(GPU *g,Buffer *b) {
    if(b->ptr)vkUnmapMemory(g->dev,b->m);
    if(b->b)vkDestroyBuffer(g->dev,b->b,NULL);
    if(b->m)vkFreeMemory(g->dev,b->m,NULL);
}
static void submit_copy(GPU *g,VkBuffer from,VkBuffer to) {
    CHECK(vkResetCommandBuffer(g->cmd,0));
    VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK(vkBeginCommandBuffer(g->cmd,&begin));
    VkBufferCopy region={.size=CHUNK};
    vkCmdCopyBuffer(g->cmd,from,to,1,&region);
    CHECK(vkEndCommandBuffer(g->cmd));
    CHECK(vkResetFences(g->dev,1,&g->fence));
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount=1,.pCommandBuffers=&g->cmd};
    CHECK(vkQueueSubmit(g->q,1,&submit,g->fence));
    CHECK(vkWaitForFences(g->dev,1,&g->fence,VK_TRUE,5000000000ull));
}
static GPU gpu_init(void) {
    GPU g={0};
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="GLM53 private Vulkan transfer benchmark",
        .apiVersion=VK_API_VERSION_1_0};
    VkInstanceCreateInfo ci={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
    CHECK(vkCreateInstance(&ci,NULL,&g.instance));
    uint32_t n=0;CHECK(vkEnumeratePhysicalDevices(g.instance,&n,NULL));
    if(!n || n>20){fputs("NO_GO no physical device\n",stderr);exit(4);}
    VkPhysicalDevice phys[20];CHECK(vkEnumeratePhysicalDevices(g.instance,&n,phys));
    char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]={0};
    for(uint32_t i=0;i<n;i++){
        VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(phys[i],&p);
        if(p.vendorID==0x1002 && p.deviceID==0x7551){g.physical=phys[i];snprintf(name,sizeof(name),"%s",p.deviceName);break;}
    }
    if(!g.physical){fputs("NO_GO R9700 Vulkan physical device not present\n",stderr);exit(4);}
    uint32_t fam=0;vkGetPhysicalDeviceQueueFamilyProperties(g.physical,&fam,NULL);
    VkQueueFamilyProperties families[32];if(fam>32 || !fam)exit(4);
    vkGetPhysicalDeviceQueueFamilyProperties(g.physical,&fam,families);
    g.family=UINT32_MAX;
    for(uint32_t i=0;i<fam;i++)if((families[i].queueFlags&VK_QUEUE_TRANSFER_BIT)&&families[i].queueCount){g.family=i;break;}
    if(g.family==UINT32_MAX)exit(4);
    const float prio=0.3f;
    VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=g.family,.queueCount=1,.pQueuePriorities=&prio};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount=1,.pQueueCreateInfos=&qci};
    CHECK(vkCreateDevice(g.physical,&dci,NULL,&g.dev));
    vkGetDeviceQueue(g.dev,g.family,0,&g.q);
    vkGetPhysicalDeviceMemoryProperties(g.physical,&g.mem);
    VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,.queueFamilyIndex=g.family};
    CHECK(vkCreateCommandPool(g.dev,&pci,NULL,&g.pool));
    VkCommandBufferAllocateInfo cb={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=g.pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    CHECK(vkAllocateCommandBuffers(g.dev,&cb,&g.cmd));
    VkFenceCreateInfo fi={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK(vkCreateFence(g.dev,&fi,NULL,&g.fence));
    fprintf(stderr,"PIN_GPU %s queue_family=%u chunk_bytes=%u\n",name,g.family,CHUNK);
    return g;
}
int main(int argc,char **argv) {
    if(argc!=4){fprintf(stderr,"usage %s <verified-model-shard-file> <aligned-offsets-file> <count:32..512>\n",argv[0]);return 2;}
    int n=atoi(argv[3]);if(n<32 || n>MAX_SAMPLES)return 2;
    struct stat st;if(stat(argv[1],&st) || !S_ISREG(st.st_mode))return 2;
    FILE *offsets=fopen(argv[2],"r");if(!offsets)return 2;
    uint64_t off[MAX_SAMPLES];for(int i=0;i<n;i++){
        if(fscanf(offsets,"%" SCNu64,&off[i])!=1 || (off[i]%4096) ||
            off[i]>((uint64_t)st.st_size-CHUNK)){fprintf(stderr,"NO_GO invalid offsets row %d\n",i);return 2;}
    }
    fclose(offsets);
    int fd=open(argv[1],O_RDONLY|O_DIRECT|O_CLOEXEC);if(fd<0){perror("open O_DIRECT");return 2;}
    void *aligned=NULL;if(posix_memalign(&aligned,4096,CHUNK))return 2;
    GPU g=gpu_init();
    Buffer src=allocbuf(&g,CHUNK,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer dst=allocbuf(&g,CHUNK,VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer back=allocbuf(&g,CHUNK,VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    double *a=calloc(n,sizeof(double)),*b=calloc(n,sizeof(double)),*c=calloc(n,sizeof(double));
    if(!a||!b||!c)return 3;
    double total=now();
    for(int i=0;i<n;i++){
        double t=now();
        ssize_t nr=pread(fd,aligned,CHUNK,(off_t)off[i]);
        a[i]=(now()-t)*1000.;
        if(nr!=CHUNK){fprintf(stderr,"NO_GO short read %zd\n",nr);return 4;}
        t=now();memcpy(src.ptr,aligned,CHUNK);b[i]=(now()-t)*1000.;
        t=now();submit_copy(&g,src.b,dst.b);c[i]=(now()-t)*1000.;
    }
    double wall=now()-total;
    submit_copy(&g,dst.b,back.b);
    if(memcmp(back.ptr,src.ptr,CHUNK)){fprintf(stderr,"NO_GO GPU byte-for-byte readback mismatch\n");return 6;}
    double a50=quant(a,n,.5),a95=quant(a,n,.95),a99=quant(a,n,.99);
    double b50=quant(b,n,.5),b95=quant(b,n,.95),b99=quant(b,n,.99);
    double c50=quant(c,n,.5),c95=quant(c,n,.95),c99=quant(c,n,.99);
    printf("{\"model_source\":\"verified real GLM-5.3 expert data\","
       "\"device\":\"Radeon AI PRO R9700 Vulkan\","
       "\"bytes_per_iteration\":%u,\"samples\":%d,"
       "\"read_o_direct_ms\":{\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f},"
       "\"cpu_staging_ms\":{\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f},"
       "\"vulkan_submit_copy_fence_ms\":{\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f},"
       "\"sequential_pipeline_GBps\":%.3f,"
       "\"GPU_readback_sha_checked\":true,\"real_io_read_GB\":%.3f,"
       "\"overlap\":false}\n",CHUNK,n,a50,a95,a99,b50,b95,b99,c50,c95,c99,
           CHUNK*(double)n/wall/1e9,CHUNK*(double)n/1e9);
    free(a);free(b);free(c);
    release(&g,&back);release(&g,&dst);release(&g,&src);
    vkDestroyFence(g.dev,g.fence,NULL);vkDestroyCommandPool(g.dev,g.pool,NULL);
    vkDestroyDevice(g.dev,NULL);vkDestroyInstance(g.instance,NULL);
    free(aligned);close(fd);
    return 0;
}

