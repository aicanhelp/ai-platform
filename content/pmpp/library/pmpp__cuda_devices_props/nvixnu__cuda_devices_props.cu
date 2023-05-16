#include "nvixnu__cuda_devices_props.h"
#include <stdio.h>

cudaDeviceProp nvixnu__get_cuda_device_props(int device_number){
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, device_number);
    return dev_prop;
}

void nvixnu__print_cuda_devices_props(){
    int dev_count, i;
    cudaDeviceProp dev_prop;

    cudaGetDeviceCount(&dev_count);
    printf("[Devices: %d]\n", dev_count);
    printf("---------------------------\n");
    for(i = 0; i < dev_count; i++){
        cudaGetDeviceProperties(&dev_prop, i);
        printf("Device [%d]:\n", i);
        printf("name: %s\n", dev_prop.name);
        printf("totalGlobalMem: %lu\n", (unsigned long)dev_prop.totalGlobalMem);
        printf("sharedMemPerMultiprocessor: %lu\n", (unsigned long)dev_prop.sharedMemPerMultiprocessor);
        printf("sharedMemPerBlock: %lu\n", (unsigned long)dev_prop.sharedMemPerBlock);
        printf("regsPerBlock: %d\n", dev_prop.regsPerBlock);
        printf("regsPerMultiprocessor: %d\n", dev_prop.regsPerMultiprocessor);
        printf("memPitch: %lu\n", (unsigned long)dev_prop.memPitch);
        printf("totalConstMem: %lu\n", (unsigned long)dev_prop.totalConstMem);
        printf("maxThreadsPerMultiProcessor: %d\n", dev_prop.maxThreadsPerMultiProcessor);
        printf("maxThreadsPerBlock: %d\n", dev_prop.maxThreadsPerBlock);
        printf("multiProcessorCount: %d\n", dev_prop.multiProcessorCount);
        printf("clockRate: %d\n", dev_prop.clockRate);
        printf("warpSize: %d\n", dev_prop.warpSize);
        printf("maxThreadsDim: (%d, %d, %d)\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
        printf("maxGridSize: (%d, %d, %d)\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
        printf("---------------------------\n\n");
    }
}