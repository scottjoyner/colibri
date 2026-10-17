#define main embedded_reference_main
#include "bench_vulkan_expert_transfer.c"
#undef main
int main(void) {
    GPU g=gpu_init();
    uint32_t n=0;CHECK(vkEnumerateDeviceExtensionProperties(g.physical,NULL,&n,NULL));
    VkExtensionProperties *props=calloc(n,sizeof(*props));
    if(!props)return 5;
    CHECK(vkEnumerateDeviceExtensionProperties(g.physical,NULL,&n,props));
    int supported=0;
    for(uint32_t i=0;i<n;i++) {
        if(strstr(props[i].extensionName,"external_memory_host") ||
           strstr(props[i].extensionName,"external_memory_dma_buf") ||
           strstr(props[i].extensionName,"host_image_copy")) {
           fprintf(stdout,"EXTENSION %s version=%u\n",props[i].extensionName,props[i].specVersion);
           if(!strcmp(props[i].extensionName,VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME))supported=1;
        }
    }
    fprintf(stdout,"HOST_EXTERNAL_MEMORY_SUPPORTED=%d extension_count=%u\n",supported,n);
    free(props);
    vkDestroyFence(g.dev,g.fence,NULL);vkDestroyCommandPool(g.dev,g.pool,NULL);
    vkDestroyDevice(g.dev,NULL);vkDestroyInstance(g.instance,NULL);
    return 0;
}
