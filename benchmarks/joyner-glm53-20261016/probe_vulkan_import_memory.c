/* Probe physical R9700 host-allocated RAM import into Vulkan, no model data. */
#define main old_reference_main
#include "bench_vulkan_expert_transfer.c"
#undef main
#define EXT VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME
typedef VkResult (VKAPI_PTR *GETPTR)(VkDevice,VkExternalMemoryHandleTypeFlagBits,const void *,VkMemoryHostPointerPropertiesEXT *);
int main(void) {
    GPU g={0};
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
       .pApplicationName="GLM53 external host probe",.apiVersion=VK_API_VERSION_1_1};
    VkInstanceCreateInfo ci={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
    CHECK(vkCreateInstance(&ci,NULL,&g.instance));
    uint32_t n=0;CHECK(vkEnumeratePhysicalDevices(g.instance,&n,NULL));
    if(!n || n>16)return 3;
    VkPhysicalDevice devices[16];CHECK(vkEnumeratePhysicalDevices(g.instance,&n,devices));
    for(uint32_t i=0;i<n;i++){
        VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(devices[i],&p);
        if(p.vendorID==0x1002&&p.deviceID==0x7551){g.physical=devices[i];break;}
    }
    if(!g.physical){fputs("NO_GO mismatched GPU\n",stderr);return 3;}
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT props_host={
       .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
       .pNext=&props_host};
    vkGetPhysicalDeviceProperties2(g.physical,&props);
    uint32_t count=0;vkGetPhysicalDeviceQueueFamilyProperties(g.physical,&count,NULL);
    if(!count || count>32)return 3;
    VkQueueFamilyProperties families[32];
    vkGetPhysicalDeviceQueueFamilyProperties(g.physical,&count,families);
    g.family=UINT32_MAX;
    for(uint32_t i=0;i<count;i++)if(families[i].queueFlags&VK_QUEUE_TRANSFER_BIT){g.family=i;break;}
    if(g.family==UINT32_MAX)return 3;
    float prio=0.25f;
    VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
       .queueFamilyIndex=g.family,.queueCount=1,.pQueuePriorities=&prio};
    const char *ext=EXT;
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
       .queueCreateInfoCount=1,.pQueueCreateInfos=&qci,
       .enabledExtensionCount=1,.ppEnabledExtensionNames=&ext};
    CHECK(vkCreateDevice(g.physical,&dci,NULL,&g.dev));
    vkGetPhysicalDeviceMemoryProperties(g.physical,&g.mem);
    GETPTR getptr=(GETPTR)vkGetDeviceProcAddr(g.dev,"vkGetMemoryHostPointerPropertiesEXT");
    if(!getptr){fputs("NO_GO missing extension function\n",stderr);return 4;}
    VkExternalMemoryBufferCreateInfo external={.sType=VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
       .handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo buf={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
       .pNext=&external,.size=CHUNK,
       .usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buffer;CHECK(vkCreateBuffer(g.dev,&buf,NULL,&buffer));
    VkMemoryRequirements req;vkGetBufferMemoryRequirements(g.dev,buffer,&req);
    VkDeviceSize align=props_host.minImportedHostPointerAlignment;
    if(align<4096)align=4096;
    if(req.alignment>align)align=req.alignment;
    void *host=NULL;
    if(posix_memalign(&host,(size_t)align,(size_t)req.size)){perror("host allocate");return 5;}
    memset(host,0,(size_t)req.size);
    VkMemoryHostPointerPropertiesEXT available={
       .sType=VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult rc=getptr(g.dev,VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,host,&available);
    printf("HOST_ALIGNMENT=%llu BUFFER_SIZE=%llu BUFFER_REQUIREMENT=%llu MEMORY_TYPE_BITS=0x%x QUERY_RESULT=%d\n",
         (unsigned long long)align,(unsigned long long)CHUNK,
         (unsigned long long)req.size,available.memoryTypeBits,rc);
    if(rc!=VK_SUCCESS){fputs("NO_GO pointer import not accepted\n",stderr);return 5;}
    uint32_t intersection=req.memoryTypeBits & available.memoryTypeBits;
    int idx=-1;
    for(uint32_t i=0;i<g.mem.memoryTypeCount;i++)
       if((intersection & (1u<<i)) &&
          ((g.mem.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
          (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
          idx=(int)i;break;
       }
    printf("MEMORY_TYPES_INTERSECTION=0x%x COHERENT_TYPE=%d\n",intersection,idx);
    if(idx<0){fputs("NO_GO no coherent imported host type\n",stderr);return 6;}
    VkImportMemoryHostPointerInfoEXT import={
       .sType=VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
       .handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,.pHostPointer=host};
    VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
       .pNext=&import,.allocationSize=req.size,.memoryTypeIndex=(uint32_t)idx};
    VkDeviceMemory mem;CHECK(vkAllocateMemory(g.dev,&ai,NULL,&mem));
    CHECK(vkBindBufferMemory(g.dev,buffer,mem,0));
    puts("PASS external host pointer import and transfer source buffer binding");
    vkDestroyBuffer(g.dev,buffer,NULL);vkFreeMemory(g.dev,mem,NULL);
    free(host);vkDestroyDevice(g.dev,NULL);vkDestroyInstance(g.instance,NULL);
    return 0;
}
